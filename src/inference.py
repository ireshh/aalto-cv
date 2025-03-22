import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import InpaintDataset
from config import config
import utils

class Predictor:
    def __init__(self, model_path):
        self.device = torch.device(config.DEVICE)
        self.model = InpaintModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def predict_batch(self, test_loader):
        preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                batch_preds = torch.sigmoid(outputs).cpu().numpy()
                preds.extend(batch_preds)
        return preds

def main():
    # Initialize
    predictor = Predictor(config.MODEL_DIR / "best_model.pth")
    test_dir = config.DATA_DIR / "test/images"
    test_paths = sorted(test_dir.glob("*.png"))
    
    # Dataset
    test_ds = InpaintDataset(test_paths, is_train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=2*config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Predict
    preds = predictor.predict_batch(test_loader)
    
    # Post-process and save
    submission = []
    for path, pred in tqdm(zip(test_paths, preds)):
        mask = utils.postprocess_mask(pred[0])
        rle = utils.mask2rle(mask)
        submission.append({
            "ImageId": path.stem,
            "EncodedPixels": rle
        })
    
    # Save CSV
    df = pd.DataFrame(submission)
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
