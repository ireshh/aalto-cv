# AaltoES 2025 Computer Vision Hackathon Solution

## Approach
- U-Net with EfficientNet-B4 encoder
- Hybrid Dice-BCE Loss
- Albumentations for augmentations
- Cosine learning rate schedule
- Test-time post-processing

## Training
```bash
python train.py
