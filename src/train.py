import torch
from dataset import InpaintingDataset
from model import create_model

def train():
    # Load config
    model = create_model()
    train_loader, val_loader = create_dataloaders()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()
    
    for epoch in range(EPOCHS):
        # Training loop
        # Validation loop
        # Save best model