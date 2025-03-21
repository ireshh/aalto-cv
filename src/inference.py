import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import InpaintingDataset
from model import create_unet
from utils import mask_to_rle
import pandas as pd

def predict():
    # Config
    test_dir = Path("../data/test/images")
    model_path = Path("../models/best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = create_unet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Dataset
    test_image_paths = sorted(test_dir.glob("*.png"))
    dataset = InpaintingDataset(test_image_paths, mode='test')
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Predict
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            outputs = torch.sigmoid(outputs)
            
            for i in range(outputs.shape[0]):
                mask = outputs[i].cpu().numpy().squeeze()
                mask = (mask > 0.5).astype(np.uint8)
                rle = mask_to_rle(mask)
                results.append(rle)
    
    # Save submission
    df = pd.DataFrame({
        "id": [path.stem for path in test_image_paths],
        "rle": results
    })
    df.to_csv("../submissions/submission.csv", index=False)

if __name__ == "__main__":
    predict()