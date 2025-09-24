# YOLO-NAS Custom Training Project

This project provides a complete pipeline for training YOLO-NAS models on custom datasets exported from CVAT.

## Features

- ✅ **Fixed bucket URLs** - Works with updated SuperGradients model weights
- ✅ **CVAT Integration** - Convert CVAT annotations to YOLO format
- ✅ **Custom Training** - Train YOLO-NAS on your own data
- ✅ **Flexible Configuration** - YAML-based configuration files
- ✅ **Apple Silicon Support** - Optimized for M1/M2 Macs with MPS

## Project Structure

```
yolo-nas/
├── data/                          # Training data
│   ├── train/                     # Training split
│   │   ├── images/               # Training images
│   │   └── labels/               # Training annotations (YOLO format)
│   ├── val/                      # Validation split
│   │   ├── images/               # Validation images
│   │   └── labels/               # Validation annotations
│   └── test/                     # Test split
│       ├── images/               # Test images
│       └── labels/               # Test annotations
├── scripts/                      # Utility scripts
│   └── convert_cvat_to_yolo.py   # CVAT to YOLO converter
├── configs/                      # Configuration files
│   ├── dataset_config.yaml       # Dataset configuration
│   └── training_config.yaml      # Training parameters
├── checkpoints/                  # Model checkpoints (created during training)
├── smoke-test.py                 # Test script for YOLO-NAS
├── train_custom_yolo_nas.py      # Main training script
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python smoke-test.py
```

### 3. Prepare Your Data

#### Export from CVAT

1. In CVAT, go to your project
2. Click "Actions" → "Export dataset"
3. Choose "CVAT 1.1" format
4. Download the ZIP file

#### Convert to YOLO Format

```bash
# Extract your CVAT export
unzip your_cvat_export.zip

# Convert to YOLO format
python scripts/convert_cvat_to_yolo.py \
    --input_cvat_xml annotations.xml \
    --input_images images/ \
    --output_dir data/all \
    --split_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1
```

### 4. Configure Your Classes

Edit `configs/dataset_config.yaml`:

```yaml
classes:
  - "person"
  - "car"
  - "bicycle"
  # Add your actual class names here

num_classes: 3  # Update to match your number of classes
```

### 5. Train Your Model

```bash
python train_custom_yolo_nas.py \
    --config configs/training_config.yaml \
    --dataset_config configs/dataset_config.yaml
```

## Configuration

### Dataset Configuration (`configs/dataset_config.yaml`)

- **classes**: List of your class names
- **input_dim**: Image size for training (default: [640, 640])
- **mixup**, **mosaic**, **copy_paste**: Data augmentation settings

### Training Configuration (`configs/training_config.yaml`)

- **model_name**: Choose from `yolo_nas_s`, `yolo_nas_m`, `yolo_nas_l`
- **max_epochs**: Number of training epochs
- **batch_size**: Batch size for training
- **initial_lr**: Initial learning rate
- **optimizer**: Optimizer type (AdamW recommended)

## Advanced Usage

### Resume Training

```bash
python train_custom_yolo_nas.py \
    --config configs/training_config.yaml \
    --dataset_config configs/dataset_config.yaml \
    --resume \
    --resume_path checkpoints/yolo_nas_custom/ckpt_best.pth
```

### Monitor Training

Training logs and checkpoints are saved in:
- `checkpoints/{experiment_name}/`
- TensorBoard logs (if enabled)

### Use Trained Model

```python
from super_gradients.training import models

# Load your trained model
model = models.get('yolo_nas_s', num_classes=3, checkpoint_path='checkpoints/yolo_nas_custom/ckpt_best.pth')

# Make predictions
predictions = model.predict('path/to/image.jpg', conf=0.25)
predictions.show()  # Display results
predictions.save('output.jpg')  # Save results
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` in training config
2. **Slow Training**: Increase `num_workers` or use smaller model (`yolo_nas_s`)
3. **Poor Performance**: Check your annotations, increase training epochs, or adjust learning rate

### Performance Tips

- **Apple Silicon**: The training automatically uses MPS (Metal Performance Shaders)
- **GPU Memory**: Start with `yolo_nas_s` and smaller batch sizes
- **Data Quality**: Ensure high-quality annotations and balanced classes

## Files Modified

This project includes fixes for SuperGradients bucket URL issues. See `buckets-for-weights.md` for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the SuperGradients license for YOLO-NAS model usage.
