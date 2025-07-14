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
        self.x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y
        # Only allow indices where a full sequence and label are available
        self.length = self.x.shape[0] - self.seq_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]  # shape: [seq_size, num_features]
        label = self.y[i+self.seq_size-1]     # label at the end of the window
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

        
    