from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloaders(d, *, test_size=0.2, batch_size=16, seed=42,
                    shuffle_train=True, num_workers=0, pin_memory=False):
    n = len(d)
    all_idx = np.arange(n)

    train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, shuffle=True)

    train_ds = Subset(d, train_idx)
    test_ds  = Subset(d, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, test_loader
