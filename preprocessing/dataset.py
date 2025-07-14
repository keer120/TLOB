import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import constants as cst
import time
from torch.utils import data
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, seq_size):
        self.seq_size = seq_size
        self.x = x
        self.y = y
        self.length = min(self.x.shape[0] - self.seq_size + 1, self.y.shape[0])
        # Debug: print shapes and length
        print(f"[Dataset] x shape: {self.x.shape}, y shape: {self.y.shape}, seq_size: {self.seq_size}, dataset length: {self.length}")
        if self.length <= 0:
            raise ValueError(f"[Dataset] Invalid dataset length: {self.length}. x shape: {self.x.shape}, y shape: {self.y.shape}, seq_size: {self.seq_size}")
        if self.x.shape[0] < self.seq_size:
            raise ValueError(f"[Dataset] x.shape[0] < seq_size: {self.x.shape[0]} < {self.seq_size}")
        if self.length != self.y.shape[0]:
            print(f"[Dataset] Warning: dataset length ({self.length}) != y.shape[0] ({self.y.shape[0]})")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i + self.seq_size > self.x.shape[0]:
            raise IndexError(f"[Dataset] i + seq_size ({i} + {self.seq_size}) exceeds x.shape[0] ({self.x.shape[0]})")
        if i >= self.y.shape[0]:
            raise IndexError(f"[Dataset] i ({i}) exceeds y.shape[0] ({self.y.shape[0]})")
        input = self.x[i:i+self.seq_size, :]
        label = self.y[i]
        return input, label
    
    # Backward compatibility for old checkpoints
    FI_2010 = "FI_2010"
    LOBSTER = "LOBSTER"
    BTC = "BTC"
    SBI = "SBI"
    COMBINED = "COMBINED"


class DataModule(pl.LightningDataModule):
    def   __init__(self, train_set, val_set, batch_size, test_batch_size,  is_shuffle_train=True, test_set=None, num_workers=16):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.is_shuffle_train = is_shuffle_train
        # Fix: use train_set.x instead of train_set.data
        if train_set.x.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.is_shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

        
    