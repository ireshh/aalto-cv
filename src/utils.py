import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import config
import torch

def get_train_paths():
    image_paths = sorted(config.TRAIN_IMAGE_DIR.glob("*.png"))
    mask_paths = sorted(config.TRAIN_MASK_DIR.glob("*.png"))
    
    print(f"Found {len(image_paths)} training images")
    print(f"Found {len(mask_paths)} training masks")
    return image_paths, mask_paths

def train_val_split(image_paths, mask_paths):
    stratify = [cv2.imread(str(p), 0).mean() > 0 for p in mask_paths]
    train_idx, val_idx = train_test_split(
        range(len(image_paths)),
        test_size=config.VAL_SPLIT,
        stratify=stratify
    )
    return (
        ([image_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx]),
        ([image_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx])
    )

def verify_data_structure():
    train_images = len(list(config.TRAIN_IMAGE_DIR.glob("*.png")))
    train_masks = len(list(config.TRAIN_MASK_DIR.glob("*.png")))
    test_images = len(list(config.TEST_IMAGE_DIR.glob("*.png")))
    
    print(f"Train images: {train_images}")
    print(f"Train masks: {train_masks}")
    print(f"Test images: {test_images}")


def evaluate(model, loader, device, criterion):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            
            preds = torch.sigmoid(outputs)
            dice = dice_coeff(preds, masks)
            dice_scores.append(dice.item())
    
    return val_loss / len(loader.dataset), np.mean(dice_scores)

def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs) if len(runs) else ''

def postprocess_mask(mask, threshold=0.5):
    binary = (mask > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
