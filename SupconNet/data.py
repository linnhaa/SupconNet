import os
import random

import numpy as np
import pandas as pd
import torch

import config


# ============================================================================
# LOADING & PRE-PROCESSING
# ============================================================================

def load_raw_data(raw_data_path, label_mapping_path):
    """
    Load raw packet CSV and reshape to (N, max_packets, 3).

    CSV layout (no header):
        Each row = one traffic sample, flattened from shape (max_packets, 4).
        Column layout per packet slot: [label, time, direction, size]
        The label is the same for every slot in a row.

    Returns
    -------
    X : np.ndarray  shape (N, max_packets, 3)  – [time, direction, size]
    y : np.ndarray  shape (N,)                 – integer class ids
    label_names : dict  {id -> name}
    max_packets : int
    """
    # Load label mapping
    label_names = {}
    if os.path.exists(label_mapping_path):
        df_map = pd.read_csv(label_mapping_path)
        label_names = dict(zip(df_map['label_id'], df_map['label_name']))
    else:
        print(f"[WARNING] Label mapping not found: {label_mapping_path}")

    # Load raw CSV and infer max_packets from column count
    raw_df = pd.read_csv(raw_data_path, header=None)
    max_packets = len(raw_df.columns) // 4

    # Reshape each row from (max_packets * 4,) → (max_packets, 4)
    # then drop the label column → features: [time, direction, size]
    all_X, all_y = [], []
    for _, row in raw_df.iterrows():
        mat = row.values.reshape(max_packets, 4)
        all_X.append(mat[:, 1:])
        all_y.append(int(mat[0, 0]))

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    print(f"[Data] Loaded {len(X)} samples, {len(np.unique(y))} classes, max_packets={max_packets}")
    return X, y, label_names, max_packets


# ============================================================================
# ORGANISE FOR CONTRASTIVE LEARNING
# ============================================================================

def build_class_list(X, y, samples_per_class=config.SAMPLES_PER_CLASS):
    """
    Group samples by class.  Removes classes with only 1 sample and caps
    each class at *samples_per_class* (random selection when over the limit).

    Returns
    -------
    data_list      : list[list[np.ndarray]]  – outer = classes, inner = samples
    filtered_labels: list[int]               – original label id per class
    """
    table = {}
    for sample, lbl in zip(X, y):
        table.setdefault(int(lbl), []).append(sample)

    data_list, filtered_labels = [], []
    for lbl, samples in table.items():
        if len(samples) < 2:
            continue
        if len(samples) > samples_per_class:
            samples = random.sample(samples, samples_per_class)
        data_list.append(samples)
        filtered_labels.append(lbl)

    total = sum(len(s) for s in data_list)
    print(f"[Data] {len(data_list)} classes, {total} total samples (cap={samples_per_class}/class)")
    return data_list, filtered_labels


# ============================================================================
# AUGMENTATION
# ============================================================================

def NetFlowAugmnet(x):
    augmented = x.clone()

    # Time jitter
    if torch.rand(1).item() > 0.2:
        noise = torch.randn_like(augmented[:, 0]) * 0.02
        augmented[:, 0] = torch.clamp(
            augmented[:, 0] + noise,
            min=0
        )

    # Size scaling
    if torch.rand(1).item() > 0.2:
        scale = 0.9 + torch.rand(1).item() * 0.2
        augmented[:, 2] = torch.round(
            augmented[:, 2] * scale
        )

    # Packet dropout
    if torch.rand(1).item() > 0.5:
        mask = (torch.rand(x.shape[0]) > 0.1).unsqueeze(1).expand_as(x).to(x.device)
        augmented = augmented * mask.float()

    # Gaussian noise
    if torch.rand(1).item() > 0.5:
    
        # Time
        augmented[:, 0] += (
            torch.randn_like(augmented[:, 0]) * 0.01
        )
    
        # Packet size
        size_noise = (
            torch.randn_like(augmented[:, 2]) * 2
        )
    
        augmented[:, 2] = torch.round(
            augmented[:, 2] + size_noise
        )
    
        augmented[:, 0] = torch.clamp(
            augmented[:, 0],
            min=0
        )
    
        augmented[:, 2] = torch.clamp(
            augmented[:, 2],
            min=0
        )

    return torch.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0)



class TwoCropTransform:
    """Applies *transform* twice independently to produce two views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


# ============================================================================
# DATASET
# ============================================================================

class PacketDataset(torch.utils.data.Dataset):
    """
    Flat dataset built from a list-of-lists (classes → samples).
    Each __getitem__ returns (views, label_idx) where views = [v1, v2]
    when a TwoCropTransform is used.
    """

    def __init__(self, data_list, transform=None):
        self.transform = transform
        self.samples, self.labels = [], []

        for label_idx, samples in enumerate(data_list):
            for sample in samples:
                if isinstance(sample, np.ndarray):
                    sample = torch.from_numpy(sample.astype(np.float32))
                self.samples.append(sample)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform is not None:
            return self.transform(sample), self.labels[idx]
        return sample, self.labels[idx]


# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def make_loader(data_list, batch_size=config.BATCH_SIZE):
    transform   = TwoCropTransform(NetFlowAugmnet)
    dataset     = PacketDataset(data_list, transform=transform)
    actual_bsz  = min(batch_size, len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = actual_bsz,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = (len(dataset) > actual_bsz),
    )
    return loader
