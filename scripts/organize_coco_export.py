#!/usr/bin/env python3
"""
Organize COCO export from CVAT for SuperGradients training.

This script takes the COCO 1.0 export from CVAT and organizes it into
the proper directory structure for training.

Usage:
    python scripts/organize_coco_export.py --input_dir path/to/cvat/coco/export --output_dir data --split_ratio 0.8 0.1 0.1
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import List, Tuple


def load_coco_json(json_path: Path) -> dict:
    """Load COCO JSON annotation file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def split_coco_dataset(coco_data: dict, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    """
    Split COCO dataset into train/val/test sets based on image IDs.
    
    Args:
        coco_data: COCO dataset dictionary
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_image_ids, val_image_ids, test_image_ids)
    """
    # Get all image IDs
    all_image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(all_image_ids)
    
    # Calculate split indices
    total_images = len(all_image_ids)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split image IDs
    train_ids = all_image_ids[:train_end]
    val_ids = all_image_ids[train_end:val_end]
    test_ids = all_image_ids[val_end:]
    
    return train_ids, val_ids, test_ids


def filter_coco_data(coco_data: dict, image_ids: List[int]) -> dict:
    """
    Filter COCO data to include only specified image IDs.
    
    Args:
        coco_data: Original COCO dataset
        image_ids: List of image IDs to keep
        
    Returns:
        Filtered COCO dataset
    """
    # Filter images
    filtered_images = [img for img in coco_data['images'] if img['id'] in image_ids]
    
    # Filter annotations (keep annotations for the filtered images)
    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    # Create filtered dataset
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories'],  # Keep all categories
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }
    
    return filtered_data


def organize_coco_export(input_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float):
    """
    Organize COCO export from CVAT into train/val/test structure.
    
    Args:
        input_dir: Directory containing CVAT COCO export
        output_dir: Output directory for organized data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
    """
    
    # Find the annotation JSON file
    annotation_files = list(input_dir.glob('**/*.json'))
    if not annotation_files:
        raise FileNotFoundError("No JSON annotation file found in the input directory")
    
    # Use the first JSON file found (should be the main annotation file)
    annotation_file = annotation_files[0]
    print(f"Using annotation file: {annotation_file}")
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco_data = load_coco_json(annotation_file)
    
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in coco_data['categories']]}")
    
    # Split dataset
    print("Splitting dataset...")
    train_ids, val_ids, test_ids = split_coco_dataset(coco_data, train_ratio, val_ratio, test_ratio)
    
    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'annotations').mkdir(exist_ok=True)
    
    # Find images directory in the export
    images_dir = None
    for subdir in input_dir.iterdir():
        if subdir.is_dir() and 'image' in subdir.name.lower():
            images_dir = subdir
            break
    
    if not images_dir:
        # Look for images in the root directory
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
        if image_files:
            images_dir = input_dir
        else:
            raise FileNotFoundError("No images directory found in the export")
    
    print(f"Using images from: {images_dir}")
    
    # Process each split
    for split_name, image_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        if not image_ids:  # Skip empty splits
            continue
            
        print(f"Processing {split_name} split...")
        
        # Create split directories
        split_dir = output_dir / split_name
        split_images_dir = split_dir / 'images'
        split_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        for img_id in image_ids:
            filename = image_id_to_filename[img_id]
            src_path = images_dir / filename
            dst_path = split_images_dir / filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
        
        # Create annotation file for this split
        filtered_data = filter_coco_data(coco_data, image_ids)
        annotation_path = split_dir / f'annotations.json'
        
        with open(annotation_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Created {split_name} split: {len(filtered_data['images'])} images, {len(filtered_data['annotations'])} annotations")
    
    # Create main annotation files (pointing to train split by default)
    main_train_json = output_dir / 'annotations' / 'instances_train.json'
    main_val_json = output_dir / 'annotations' / 'instances_val.json'
    
    if train_ids:
        train_data = filter_coco_data(coco_data, train_ids)
        with open(main_train_json, 'w') as f:
            json.dump(train_data, f, indent=2)
    
    if val_ids:
        val_data = filter_coco_data(coco_data, val_ids)
        with open(main_val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
    
    # Copy all images to main images directory
    print("Copying all images to main images directory...")
    for img_file in images_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy2(img_file, output_dir / 'images' / img_file.name)
    
    print("Organization completed!")
    print(f"Output structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/          # All images")
    print(f"  ├── annotations/     # Main annotation files")
    print(f"  ├── train/           # Training split")
    print(f"  ├── val/             # Validation split")
    print(f"  └── test/            # Test split")


def main():
    parser = argparse.ArgumentParser(description='Organize COCO export from CVAT')
    parser.add_argument('--input_dir', required=True, help='Directory containing CVAT COCO export')
    parser.add_argument('--output_dir', required=True, help='Output directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed
    random.seed(args.seed)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Organize the export
    organize_coco_export(input_dir, output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == '__main__':
    main()
