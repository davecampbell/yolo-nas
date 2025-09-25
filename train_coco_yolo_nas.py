#!/usr/bin/env python3
"""
Train YOLO-NAS model on COCO format dataset from CVAT.

This script trains a YOLO-NAS model using SuperGradients on a dataset
exported from CVAT in COCO 1.0 format.

Usage:
    python train_coco_yolo_nas.py --config configs/training_config.yaml --dataset_config configs/coco_dataset_config.yaml
"""

import argparse
import yaml
import torch
import json
from pathlib import Path
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> str:
    """Setup device for training (MPS, CUDA, or CPU)."""
    if config.get('device') == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA GPU")
        else:
            device = 'cpu'
            print("Using CPU")
    else:
        device = config['device']
        print(f"Using specified device: {device}")
    
    return device


def extract_classes_from_coco_json(json_path: str) -> list:
    """Extract class names from COCO JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Sort categories by id to maintain consistent ordering
    categories = sorted(data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    
    return class_names


def create_coco_dataloaders(dataset_config: dict, training_config: dict):
    """Create COCO training and validation dataloaders."""
    
    # Extract class names from COCO JSON if not provided
    if not dataset_config.get('classes'):
        print("Extracting class names from COCO JSON...")
        train_json_path = Path(dataset_config['data_dir']) / dataset_config['train_json_file']
        dataset_config['classes'] = extract_classes_from_coco_json(str(train_json_path))
        print(f"Found classes: {dataset_config['classes']}")
    
    # Training dataset
    train_dataset = COCODetectionDataset(
        data_dir=dataset_config['data_dir'],
        subdir=dataset_config['train_images_dir'],
        json_file=dataset_config['train_json_file'],
        input_dim=dataset_config['input_dim'],
        mixup=dataset_config.get('mixup', 0.0),
        copy_paste=dataset_config.get('copy_paste', 0.0),
        mosaic=dataset_config.get('mosaic', 1.0),
        mosaic_prob=dataset_config.get('mosaic_prob', 1.0),
        mixup_prob=dataset_config.get('mixup_prob', 1.0),
        mixup_alpha=dataset_config.get('mixup_alpha', 8.0),
        degrees=dataset_config.get('degrees', 10.0),
        translate=dataset_config.get('translate', 0.1),
        scale=dataset_config.get('scale', [0.1, 2.0]),
        shear=dataset_config.get('shear', 2.0),
        perspective=dataset_config.get('perspective', 0.0),
        flip_ud=dataset_config.get('flip_ud', 0.0),
        flip_lr=dataset_config.get('flip_lr', 0.5),
        hsv_h=dataset_config.get('hsv_h', 0.015),
        hsv_s=dataset_config.get('hsv_s', 0.7),
        hsv_v=dataset_config.get('hsv_v', 0.4),
        cache_images=dataset_config.get('cache_images', False),
        cache_format=dataset_config.get('cache_format', 'ram'),
        rect_training=dataset_config.get('rect_training', False),
    )
    
    # Validation dataset
    val_dataset = COCODetectionDataset(
        data_dir=dataset_config['data_dir'],
        subdir=dataset_config['val_images_dir'],
        json_file=dataset_config['val_json_file'],
        input_dim=dataset_config['input_dim'],
        cache_images=dataset_config.get('cache_images', False),
        cache_format=dataset_config.get('cache_format', 'ram'),
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )
    
    return train_dataloader, val_dataloader


def create_model(training_config: dict, dataset_config: dict):
    """Create YOLO-NAS model."""
    
    model = models.get(
        training_config['model_name'],
        num_classes=len(dataset_config['classes']),
        pretrained_weights=training_config['pretrained_weights']
    )
    
    return model


def create_metrics():
    """Create metrics for evaluation."""
    return [
        DetectionMetrics_050(score_thres=0.1, top_k_predictions=300),
        DetectionMetrics_050_095(score_thres=0.1, top_k_predictions=300)
    ]


def create_training_params(training_config: dict):
    """Create training parameters."""
    
    training_params = {
        'max_epochs': training_config['max_epochs'],
        'lr_mode': training_config['lr_mode'],
        'initial_lr': training_config['initial_lr'],
        'cosine_final_lr_ratio': training_config['cosine_final_lr_ratio'],
        'warmup_epochs': training_config['warmup_epochs'],
        'warmup_initial_lr': training_config['warmup_initial_lr'],
        'loss': training_config['loss'],
        'optimizer': training_config['optimizer'],
        'optimizer_params': training_config['optimizer_params'],
        'train_metrics_list': training_config['train_metrics_list'],
        'valid_metrics_list': training_config['valid_metrics_list'],
        'metric_to_watch': training_config['metric_to_watch'],
        'greater_metric_to_watch_is_better': training_config['greater_metric_to_watch_is_better'],
        'save_ckpt_epoch_list': training_config['save_ckpt_epoch_list'],
        'ckpt_root_dir': training_config['ckpt_root_dir'],
        'experiment_name': training_config['experiment_name'],
        'log_instances': training_config.get('log_instances', True),
        'log_images': training_config.get('log_images', False),
        'log_images_interval': training_config.get('log_images_interval', 50),
        'run_validation_freq': training_config.get('run_validation_freq', 1),
        'mixed_precision': training_config.get('mixed_precision', True),
        'resume': training_config.get('resume', False),
        'resume_path': training_config.get('resume_path', ''),
    }
    
    # Add early stopping if enabled
    if training_config.get('early_stop', False):
        training_params['early_stop'] = True
        training_params['early_stop_patience'] = training_config.get('early_stop_patience', 10)
    
    return training_params


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-NAS on COCO format dataset')
    parser.add_argument('--config', required=True, help='Path to training configuration file')
    parser.add_argument('--dataset_config', required=True, help='Path to COCO dataset configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    training_config = load_config(args.config)
    dataset_config = load_config(args.dataset_config)
    
    # Override resume settings if provided via command line
    if args.resume:
        training_config['resume'] = True
    if args.resume_path:
        training_config['resume_path'] = args.resume_path
    
    # Setup device
    device = setup_device(training_config)
    
    # Create dataloaders
    print("Creating COCO dataloaders...")
    train_dataloader, val_dataloader = create_coco_dataloaders(dataset_config, training_config)
    
    # Create model
    print(f"Creating {training_config['model_name']} model...")
    model = create_model(training_config, dataset_config)
    
    # Create metrics
    metrics = create_metrics()
    
    # Create training parameters
    training_params = create_training_params(training_config)
    training_params['valid_metrics_list'] = metrics
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        experiment_name=training_config['experiment_name'],
        ckpt_root_dir=training_config['ckpt_root_dir']
    )
    
    # Start training
    print("Starting training...")
    print(f"Dataset: {len(train_dataloader.dataset)} training images, {len(val_dataloader.dataset)} validation images")
    print(f"Classes: {dataset_config['classes']}")
    print(f"Training for {training_config['max_epochs']} epochs...")
    
    trainer.train(
        model=model,
        training_params=training_params,
        train_loader=train_dataloader,
        valid_loader=val_dataloader
    )
    
    print("Training completed!")
    print(f"Checkpoints saved in: {training_config['ckpt_root_dir']}/{training_config['experiment_name']}")


if __name__ == '__main__':
    main()
