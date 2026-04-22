import os

import numpy as np
import torch

from .. import model
from model import SupConNet


def load_encoder(model_path: str) -> SupConNet | None:
    """
    Load a SupConNet checkpoint and return the model in eval mode.
    Returns None if the checkpoint file does not exist.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    ckpt   = torch.load(model_path, map_location='cpu')
    cfg    = ckpt.get('model_config', {})
    model  = SupConNet(
        max_packets    = cfg.get('max_packets',    15000),
        hidden_size    = cfg.get('hidden_size',    256),
        embedding_size = cfg.get('embedding_size', 128),
        head           = 'mlp',
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    print(f"[Model] Loaded from {model_path} (epoch {ckpt.get('epoch', '?')}) on {device}")
    return model


def extract_embeddings(model: SupConNet, X_raw: np.ndarray,
                       batch_size: int = 32) -> np.ndarray:
    encoder = model.encoder
    encoder.eval()
    device  = next(model.parameters()).device
    parts   = []

    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            batch = torch.FloatTensor(X_raw[i : i + batch_size]).to(device)
            parts.append(encoder(batch).cpu().numpy())

    return np.vstack(parts)