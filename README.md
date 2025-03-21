# AI Inpainting Detection

## Approach
- U-Net with EfficientNet-B4 encoder
- Combined Dice + BCE loss
- Image augmentations with Albumentations
- Learning rate scheduling

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train: `python src/train.py`
3. Predict: `python src/inference.py`

HELLO WORLD