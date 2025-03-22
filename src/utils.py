import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import config

# In utils.py update get_train_paths()
def get_train_paths(data_dir):
    image_dir = data_dir / "train/images"
    mask_dir = data_dir / "train/masks"
    
    print(f"Image dir exists: {image_dir.exists()}")
    print(f"Mask dir exists: {mask_dir.exists()}")
    print(f"Sample image: {list(image_dir.glob('*.png'))[:3]}")
    
    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    return image_paths, mask_paths

def train_val_split(image_paths, mask_paths, test_size=0.2):
    stratify = [cv2.imread(str(p), 0).mean() > 0 for p in mask_paths]
    train_idx, val_idx = train_test_split(
        range(len(image_paths)),
        test_size=test_size,
        stratify=stratify
    )
    return (
        ([image_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx]),
        ([image_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx])
    )

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
