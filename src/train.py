import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from dataset import InpaintDataset
from model import InpaintModel, HybridLoss
import utils

def main():
    torch.manual_seed(42)
    device = config.DEVICE
    config.MODEL_DIR.mkdir(exist_ok=True)
    
    image_paths, mask_paths = utils.get_train_paths()
    train_paths, val_paths = utils.train_val_split(image_paths, mask_paths)
    
    train_ds = InpaintDataset(*train_paths)
    val_ds = InpaintDataset(*val_paths, is_train=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    model = InpaintModel().to(device)
    criterion = HybridLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=2
    )
    writer = SummaryWriter()
    
    best_dice = 0
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=loss.item())
        
        val_loss, val_dice = utils.evaluate(model, val_loader, device, criterion)
        scheduler.step(val_dice)

        writer.add_scalar("Loss/train", train_loss / len(train_ds), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config.MODEL_DIR / "best_model.pth")
        
        print(f"Epoch {epoch+1}: Val Dice = {val_dice:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()