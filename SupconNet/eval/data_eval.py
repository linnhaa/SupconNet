import os

import numpy as np
import pandas as pd


def load_raw_packets(data_path: str, read_labels: bool = True):
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        return None, None

    raw_df      = pd.read_csv(data_path, header=None)
    max_packets = len(raw_df.columns) // 4

    all_X, all_y = [], []
    for _, row in raw_df.iterrows():
        mat = row.values.reshape(max_packets, 4)
        all_X.append(mat[:, 1:].astype(np.float32))   # drop label column
        if read_labels:
            all_y.append(int(mat[0, 0]))

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64) if read_labels else None

    n_cls = len(np.unique(y)) if y is not None else '?'
    print(f"[Data] {data_path}  →  {X.shape},  {n_cls} classes")
    return X, y


def load_label_names(label_mapping_path: str, fallback_ids=()) -> dict:
    names = {int(i): f"class_{i}" for i in fallback_ids}
    if label_mapping_path and os.path.exists(label_mapping_path):
        df = pd.read_csv(label_mapping_path)
        if {'label_id', 'label_name'}.issubset(df.columns):
            for _, row in df.iterrows():
                names[int(row['label_id'])] = str(row['label_name'])
    return names