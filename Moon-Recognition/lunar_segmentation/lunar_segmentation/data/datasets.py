import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from pathlib import Path

class MoonTileDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, augment: bool = False):
        self.index_df = index_df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.index_df)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        # Horizontal Flip
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
        # Vertical Flip
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        # Random Rotation (90, 180, 270)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return image, mask

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        data = np.load(row['tile_path'])
        image = data['image'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)
        return torch.from_numpy(image), torch.from_numpy(mask)
