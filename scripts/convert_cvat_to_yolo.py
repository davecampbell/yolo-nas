#!/usr/bin/env python3
"""
Convert CVAT annotations to YOLO format for SuperGradients training.

This script handles CVAT XML exports and converts them to YOLO format
required by SuperGradients YOLO-NAS training.

Usage:
    python scripts/convert_cvat_to_yolo.py --input_cvat_xml path/to/annotations.xml --input_images path/to/images --output_dir data/train
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import json


def parse_cvat_xml(xml_path: str) -> Tuple[Dict[str, List], Dict[int, str]]:
    """
    Parse CVAT XML file and extract annotations.
    
    Args:
        xml_path: Path to CVAT XML annotation file
        
    Returns:
        Tuple of (annotations_dict, class_mapping)
        annotations_dict: {image_name: [annotation_data]}
        class_mapping: {class_id: class_name}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    class_mapping = {}
    
    # Extract class information
    for label in root.findall('.//label'):
        class_id = int(label.get('id'))
        class_name = label.get('name')
        class_mapping[class_id] = class_name
    
    # Extract image annotations
    for image in root.findall('.//image'):
        image_name = image.get('name')
        image_width = int(image.get('width'))
        image_height = int(image.get('height'))
        
        image_annotations = []
        
        for box in image.findall('.//box'):
            class_id = int(box.get('label'))
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr'))
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Adjust class_id to be 0-based for YOLO format
            yolo_class_id = class_id - 1
            
            image_annotations.append({
                'class_id': yolo_class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        if image_annotations:  # Only include images with annotations
            annotations[image_name] = {
                'annotations': image_annotations,
                'width': image_width,
                'height': image_height
            }
    
    return annotations, class_mapping


def convert_to_yolo_format(annotations: Dict[str, List], output_dir: Path, class_mapping: Dict[int, str]):
    """
    Convert annotations to YOLO format and save files.
    
    Args:
        annotations: Parsed annotations dictionary
        output_dir: Output directory for YOLO format files
        class_mapping: Mapping of class IDs to class names
    """
    # Create output directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Save class names file
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        for class_id in sorted(class_mapping.keys()):
            f.write(f"{class_mapping[class_id]}\n")
    
    # Save class mapping as JSON for reference
    class_mapping_file = output_dir / 'class_mapping.json'
    with open(class_mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Process each image
    for image_name, data in annotations.items():
        # Create YOLO annotation file
        label_file = labels_dir / f"{Path(image_name).stem}.txt"
        
        with open(label_file, 'w') as f:
            for annotation in data['annotations']:
                f.write(f"{annotation['class_id']} "
                       f"{annotation['x_center']:.6f} "
                       f"{annotation['y_center']:.6f} "
                       f"{annotation['width']:.6f} "
                       f"{annotation['height']:.6f}\n")
        
        print(f"Created annotation: {label_file}")


def copy_images(input_images_dir: Path, output_dir: Path, image_names: List[str]):
    """
    Copy images to output directory.
    
    Args:
        input_images_dir: Directory containing input images
        output_dir: Output directory
        image_names: List of image names to copy
    """
    images_output_dir = output_dir / 'images'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name in image_names:
        src_path = input_images_dir / image_name
        dst_path = images_output_dir / image_name
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied image: {image_name}")
        else:
            print(f"Warning: Image not found: {src_path}")


def split_dataset(data_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Directory containing all data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    import random
    
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(image_files)
    
    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split images
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]
    
    # Move files to appropriate directories
    for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        split_dir = data_dir.parent / split_name
        split_images_dir = split_dir / 'images'
        split_labels_dir = split_dir / 'labels'
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in split_images:
            # Copy image
            shutil.copy2(image_path, split_images_dir / image_path.name)
            
            # Copy corresponding label
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, split_labels_dir / f"{image_path.stem}.txt")
    
    print(f"Dataset split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")


def main():
    parser = argparse.ArgumentParser(description='Convert CVAT annotations to YOLO format')
    parser.add_argument('--input_cvat_xml', required=True, help='Path to CVAT XML annotation file')
    parser.add_argument('--input_images', required=True, help='Path to directory containing images')
    parser.add_argument('--output_dir', required=True, help='Output directory for YOLO format data')
    parser.add_argument('--split_dataset', action='store_true', help='Split dataset into train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Parse CVAT XML
    print(f"Parsing CVAT XML: {args.input_cvat_xml}")
    annotations, class_mapping = parse_cvat_xml(args.input_cvat_xml)
    
    print(f"Found {len(annotations)} annotated images")
    print(f"Classes: {list(class_mapping.values())}")
    
    # Convert to YOLO format
    output_dir = Path(args.output_dir)
    convert_to_yolo_format(annotations, output_dir, class_mapping)
    
    # Copy images
    input_images_dir = Path(args.input_images)
    copy_images(input_images_dir, output_dir, list(annotations.keys()))
    
    # Split dataset if requested
    if args.split_dataset:
        print("Splitting dataset...")
        split_dataset(output_dir, args.train_ratio, args.val_ratio)
    
    print("Conversion completed!")


if __name__ == '__main__':
    main()
