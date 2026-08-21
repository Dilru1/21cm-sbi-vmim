"""Torch-only DataLoader helper for the NLE stage (summaries in memory)."""
import torch
from torch.utils.data import DataLoader, TensorDataset

def make_loader(theta, t, batch_size, shuffle, num_workers, pin):
    return DataLoader(TensorDataset(torch.tensor(theta, dtype=torch.float32),
                                    torch.tensor(t, dtype=torch.float32)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin, persistent_workers=(num_workers > 0))
