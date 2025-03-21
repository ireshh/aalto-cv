import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import InpaintingDataset
from model import create_unet, DiceBCELoss
from utils import calculate_dice
import torch.nn as nn

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    image_dir = Path(config['data_path']) / "train/manipulated"
    mask_dir = Path(config['data_path']) / "train/masks"
    
    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    
    # Train/Validation split
    dataset = InpaintingDataset(image_paths, mask_paths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model setup
    model = create_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    criterion = DiceBCELoss()
    
    best_dice = 0
    early_stop_counter = 0

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        # Training phase
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs)) > 0.5
                val_dice += calculate_dice(preds, masks).item()
        
        # Calculate metrics
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)
        
        # Update scheduler
        scheduler.step(val_dice)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            early_stop_counter = 0
            torch.save(model.state_dict(), Path(config['model_save_path']) / "best_model.pth")
            print(f"New best model saved with Dice: {best_dice:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    main()