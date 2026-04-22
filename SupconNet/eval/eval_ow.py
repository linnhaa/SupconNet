import os

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.neural_network import MLPClassifier

from data_eval import load_label_names, load_raw_packets
from embed import extract_embeddings, load_encoder

# ============================================================================
# CONFIGURATION
# ============================================================================

KNOWN_TRAIN_PATH   = 'data_pretrain/50labels500_mac_2cdns/embedding_data/raw_packet_data.csv'
KNOWN_TEST_PATH    = 'data_pretrain/50labels500_mac_2cdns/test_data/raw_packet_data.csv'
UNKNOWN_DATA_PATH  = 'data_pretrain/50labels500_mac_new/embedding_data/raw_packet_data.csv'
LABEL_MAPPING_PATH = 'data_pretrain/50labels500_mac/embedding_data/label_mapping.csv'
MODEL_PATH         = 'model/augment/supconnet.pth'
OUTPUT_FOLDER      = 'output/open_world'

# Class id assigned to all unknown traffic (must be outside the known id range)
UNKNOWN_CLASS_ID = 50

# How many unknown samples to mix into training / test sets
UNKNOWN_TRAIN_SAMPLES = 400
UNKNOWN_TEST_SAMPLES  = 10000   # set to None to use all remaining samples

# ============================================================================
# DATA HELPERS (open-world specific)
# ============================================================================

def _split_unknown_pool(unknown_path: str, train_n: int, test_n: int | None,
                        unknown_id: int):
    X_unk, _ = load_raw_packets(unknown_path, read_labels=False)
    if X_unk is None:
        return None, None, None, None

    rng  = np.random.default_rng(seed=42)
    X_unk = X_unk[rng.permutation(len(X_unk))]

    total        = len(X_unk)
    test_n_actual = (total - train_n) if test_n is None else test_n

    if train_n + test_n_actual > total:
        print(f"[WARNING] Requested {train_n + test_n_actual} unknown samples "
              f"but only {total} available. Adjusting.")
        train_n       = min(train_n, total)
        test_n_actual = total - train_n

    X_tr = X_unk[:train_n]
    X_te = X_unk[train_n : train_n + test_n_actual]
    y_tr = np.full(len(X_tr), unknown_id, dtype=np.int64)
    y_te = np.full(len(X_te), unknown_id, dtype=np.int64)

    print(f"[Data] Unknown pool split — train: {len(y_tr)},  test: {len(y_te)}")
    return X_tr, y_tr, X_te, y_te


def _shuffle(X, y, seed=0):
    perm = np.random.default_rng(seed).permutation(len(X))
    return X[perm], y[perm]

# ============================================================================
# REPORTING HELPERS
# ============================================================================

def _save_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)


def _print_binary_cm(cm, tn, fp, fn, tp):
    """Print the Known-vs-Unknown 2×2 summary."""
    cm_df = pd.DataFrame(cm,
                         index   = ['Actual Unknown', 'Actual Known'],
                         columns = ['Pred Unknown',   'Pred Known'])
    print(f"\n[Confusion Matrix – Known vs Unknown]\n{cm_df.to_string()}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return cm_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"[Config] known classes=0..{UNKNOWN_CLASS_ID - 1}  "
          f"unknown_id={UNKNOWN_CLASS_ID}  "
          f"unk_train={UNKNOWN_TRAIN_SAMPLES}  "
          f"unk_test={UNKNOWN_TEST_SAMPLES or 'ALL'}")

    # 1. Load encoder
    encoder = load_encoder(MODEL_PATH)
    if encoder is None:
        return

    # 2. Known train / test
    X_kn_tr, y_kn_tr = load_raw_packets(KNOWN_TRAIN_PATH)
    X_kn_te, y_kn_te = load_raw_packets(KNOWN_TEST_PATH)
    if X_kn_tr is None or X_kn_te is None:
        return

    # 3. Unknown pool → split
    X_uk_tr, y_uk_tr, X_uk_te, y_uk_te = _split_unknown_pool(
        UNKNOWN_DATA_PATH, UNKNOWN_TRAIN_SAMPLES,
        UNKNOWN_TEST_SAMPLES, UNKNOWN_CLASS_ID,
    )
    if X_uk_tr is None:
        return

    # 4. Merge & shuffle
    X_tr_raw, y_tr = _shuffle(
        np.concatenate([X_kn_tr, X_uk_tr]),
        np.concatenate([y_kn_tr, y_uk_tr]),
    )
    X_te_raw, y_te = _shuffle(
        np.concatenate([X_kn_te, X_uk_te]),
        np.concatenate([y_kn_te, y_uk_te]),
    )
    print(f"[Data] Final — train: {len(y_tr)},  test: {len(y_te)}")

    # 5. Label names
    all_known_ids = np.unique(np.concatenate([y_kn_tr, y_kn_te]))
    id_to_name    = load_label_names(LABEL_MAPPING_PATH, fallback_ids=all_known_ids)
    id_to_name[UNKNOWN_CLASS_ID] = 'Unknown'

    # 6. Embeddings
    print("[Embed] Extracting embeddings …")
    X_tr = extract_embeddings(encoder, X_tr_raw)
    X_te = extract_embeddings(encoder, X_te_raw)

    # 7. Train MLP
    print("[Train] Fitting MLP …")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), activation='relu',
        solver='adam', max_iter=200, early_stopping=True,
        random_state=94, verbose=False,
    )
    clf.fit(X_tr, y_tr)

    # 8. Predict
    y_pred = clf.predict(X_te)
    proba  = clf.predict_proba(X_te)

    # 9. Per-class report
    all_ids      = sorted(np.unique(np.concatenate([y_tr, y_te])))
    target_names = [id_to_name.get(i, f"class_{i}") for i in all_ids]
    report       = classification_report(y_te, y_pred, labels=all_ids,
                                         target_names=target_names,
                                         digits=4, zero_division=0)
    print(f"\n[Report]\n{report}")

    # 10. Binary (known vs unknown) confusion matrix
    known_mask = clf.classes_ != UNKNOWN_CLASS_ID
    score_known = proba[:, known_mask].max(axis=1)

    y_bin_te   = (y_te   != UNKNOWN_CLASS_ID).astype(int)
    y_bin_pred = (y_pred != UNKNOWN_CLASS_ID).astype(int)

    cm_bin           = confusion_matrix(y_bin_te, y_bin_pred, labels=[0, 1])
    tn, fp, fn, tp   = cm_bin.ravel()
    cm_bin_df        = _print_binary_cm(cm_bin, tn, fp, fn, tp)

    # 11. Full confusion matrix
    cm_full    = confusion_matrix(y_te, y_pred, labels=all_ids)
    cm_full_df = pd.DataFrame(cm_full, index=target_names, columns=target_names)

    # 12. PR / ROC curves
    y_true_bin = (y_te != UNKNOWN_CLASS_ID).astype(int)
    precision, recall, _  = precision_recall_curve(y_true_bin, score_known)
    fpr,       tpr,    _  = roc_curve(y_true_bin, score_known)

    # 13. Per-sample prediction table (with class probabilities)
    proba_df = pd.DataFrame(
        proba,
        columns=[f"prob_{id_to_name.get(c, c)}" for c in clf.classes_]
    )
    pred_df = pd.concat([
        pd.DataFrame({
            'y_true':      y_te,
            'y_true_name': [id_to_name.get(v, f"class_{v}") for v in y_te],
            'y_pred':      y_pred,
            'y_pred_name': [id_to_name.get(v, f"class_{v}") for v in y_pred],
        }),
        proba_df,
    ], axis=1)

    # 14. Save all outputs
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    curve_dir = os.path.join(OUTPUT_FOLDER, 'curve_data')
    os.makedirs(curve_dir, exist_ok=True)

    header = (
        "=" * 60 + "\nOPEN-WORLD MULTI-CLASS CLASSIFICATION REPORT\n"
        f"  known classes : 0..{UNKNOWN_CLASS_ID - 1}\n"
        f"  unknown class : {UNKNOWN_CLASS_ID}\n"
        f"  unknown train : {UNKNOWN_TRAIN_SAMPLES} samples\n"
        f"  unknown test  : {UNKNOWN_TEST_SAMPLES} samples\n"
        + "=" * 60 + "\n\n"
    )
    _save_text(
        os.path.join(OUTPUT_FOLDER, 'classification_report.txt'),
        header + report
        + "\n\n" + "=" * 60 + "\nCONFUSION MATRIX (Known vs Unknown)\n" + "=" * 60
        + f"\n\n{cm_bin_df.to_string()}\n\nTN={tn}  FP={fp}  FN={fn}  TP={tp}\n",
    )
    cm_full_df.to_csv(os.path.join(OUTPUT_FOLDER, 'confusion_matrix_full.csv'))
    cm_bin_df.to_csv(os.path.join(OUTPUT_FOLDER,  'confusion_matrix_summary.csv'))
    pred_df.to_csv(os.path.join(OUTPUT_FOLDER,    'predictions.csv'), index=False)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(
        os.path.join(curve_dir, 'precision_recall_curve.csv'), index=False)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
        os.path.join(curve_dir, 'roc_curve.csv'), index=False)

    print(f"[Done] All outputs saved to {OUTPUT_FOLDER}/")


if __name__ == '__main__':
    main()