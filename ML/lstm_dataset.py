import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, time_steps=10):
    """
    Converts 2D data (Samples, Features) into 3D sequences (Samples, TimeSteps, Features).
    
    Args:
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        time_steps (int): Number of lookback steps.
        
    Returns:
        X_seq (np.array): (N - time_steps + 1, time_steps, F)
        y_seq (np.array): (N - time_steps + 1, )
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps - 1]) # Target is the value at the end of the sequence
    
    return np.array(Xs), np.array(ys)

def prepare_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = TimeSeriesDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
