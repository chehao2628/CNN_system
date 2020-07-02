import numpy as np
from preprocessing import *
import torch


def load_data_kfold(batch_size, k, n):
    # This function using functions in preprocessing.py to build dataset,
    # and then randomly split dataset with a fixed random seed.
    print("Building DataSet ...")
    create_path()
    data = build_dataset()  # Build data set
    print("Complete.")

    print("Splitting DataSet ...")

    l = len(data)
    print(l)
    shuffle_dataset = True
    random_seed = 42  # fixed random seed
    indices = list(range(l))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)  # shuffle
    # Collect indexes of samples for validation set.
    val_indices = indices[int(l / k) * n:int(l / k) * (n + 1)]
    # Collect indexes of samples for train set. Here the logic is that a sample
    # cannot in train set if already in validation set
    train_indices = list(set(indices).difference(set(val_indices)))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)  # build Sampler
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler)  # build dataloader for train set
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler)  # build dataloader for validate set
    print("Complete.")
    return train_loader, validation_loader


def load_data(batch_size):
    print("Building DataSet ...")
    data = build_dataset()  # Build data set
    print("Complete.")

    print("Splitting DataSet ...")
    validation_split = 0.1
    l = len(data)
    print(l)
    shuffle_dataset = True
    random_seed = 42
    indices = list(range(l))
    split = int(np.floor(validation_split * l))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
    print()
    print("Complete.")
    return train_loader, validation_loader
