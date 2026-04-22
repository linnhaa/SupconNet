import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from sklearn.neural_network import MLPClassifier

from .data_eval import load_label_names, load_raw_packets
from .embed import extract_embeddings, load_encoder


def _subsample_per_class(X: np.ndarray, y: np.ndarray,
                         n: int | None, seed: int = 0) -> tuple:
    if n is None:
        return X, y
    rng   = np.random.default_rng(seed)
    keep  = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    idx_all = np.sort(np.concatenate(keep))
    return X[idx_all], y[idx_all]

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_DATA_PATH    = 'data_pretrain/50labels500_mac_2cdns/embedding_data/raw_packet_data.csv'
TEST_DATA_PATH     = 'data_pretrain/50labels500_mac_2cdns/test_data/raw_packet_data.csv'
LABEL_MAPPING_PATH = 'data_pretrain/50labels500_mac_2cdns/embedding_data/label_mapping.csv'
MODEL_PATH         = 'model/supconnet.pth'
OUTPUT_FOLDER      = 'output/closed_world'

# Max training samples per class (None = use all available)
TRAIN_SAMPLES_PER_CLASS = 200

# ============================================================================
# CLASSIFIER
# ============================================================================

MODELS = {
    'MLP': MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), activation='relu',
        solver='adam', max_iter=200, early_stopping=True,
        random_state=94, verbose=False),
}

# ============================================================================
# TRAIN & EVALUATE ONE CLASSIFIER
# ============================================================================

def _run_one(name, clf, X_tr, y_tr, X_te, y_te,
             unique_labels, label_names, out_dir):
    t0     = time.time()
    clf.fit(X_tr, y_tr)
    t_fit  = time.time() - t0

    t1     = time.time()
    y_pred = clf.predict(X_te)
    t_inf  = time.time() - t1

    acc  = accuracy_score(y_te, y_pred)
    mf1  = f1_score(y_te, y_pred, average='macro',    zero_division=0)
    wf1  = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print(f"  [{name}]  acc={acc:.4f}  macro_f1={mf1:.4f}  "
          f"weighted_f1={wf1:.4f}  train={t_fit:.1f}s  inf={t_inf:.2f}s")

    # per-model text report
    target_names = [label_names.get(l, f'class_{l}') for l in unique_labels]
    report = classification_report(y_te, y_pred, labels=unique_labels,
                                   target_names=target_names,
                                   digits=4, zero_division=0)
    os.makedirs(out_dir, exist_ok=True)
    rpath = os.path.join(out_dir, f"{name.replace(' ', '_')}_report.txt")
    with open(rpath, 'w') as fh:
        fh.write(f"MODEL: {name}\n"
                 f"train_time={t_fit:.1f}s  inf_time={t_inf:.2f}s\n\n"
                 + "=" * 60 + "\nCLASSIFICATION REPORT\n" + "=" * 60 + "\n\n"
                 + report)

    return dict(model=name,
                accuracy=round(acc, 4),
                macro_f1=round(mf1, 4),
                weighted_f1=round(wf1, 4),
                train_time_s=round(t_fit, 1),
                inf_time_s=round(t_inf, 2))

# ============================================================================
# MAIN
# ============================================================================

def main():
    # 1. Load encoder
    encoder = load_encoder(MODEL_PATH)
    if encoder is None:
        return

    # 2. Load raw data
    X_tr_raw, y_tr = load_raw_packets(TRAIN_DATA_PATH)
    X_te_raw, y_te = load_raw_packets(TEST_DATA_PATH)
    if X_tr_raw is None or X_te_raw is None:
        return

    # 3. Extract embeddings
    print("[Embed] Extracting train embeddings …")
    X_tr = extract_embeddings(encoder, X_tr_raw)
    print("[Embed] Extracting test  embeddings …")
    X_te = extract_embeddings(encoder, X_te_raw)

    # 4. Subsample training set
    X_tr, y_tr = _subsample_per_class(X_tr, y_tr, TRAIN_SAMPLES_PER_CLASS)
    print(f"[Data] Training after subsample: {len(y_tr)} samples "
          f"(cap={TRAIN_SAMPLES_PER_CLASS}/class)")

    # 4. Label names
    unique_labels = np.unique(np.concatenate([y_tr, y_te]))
    label_names   = load_label_names(LABEL_MAPPING_PATH, fallback_ids=unique_labels)

    # 5. Benchmark
    print(f"\n[Benchmark] Running {len(MODELS)} classifiers …")
    rows = []
    for name, clf in MODELS.items():
        try:
            rows.append(_run_one(name, clf, X_tr, y_tr, X_te, y_te,
                                 unique_labels, label_names, OUTPUT_FOLDER))
        except Exception as exc:
            print(f"  [WARNING] {name} failed: {exc}")

    # 6. Summary table
    summary = (pd.DataFrame(rows)
               .sort_values('weighted_f1', ascending=False)
               .reset_index(drop=True))
    summary.index += 1
    print(f"\n[Summary]\n{summary.to_string()}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, 'model_comparison.csv')
    summary.to_csv(out_path)
    print(f"[Done] Summary saved to {out_path}")


if __name__ == '__main__':
    main()