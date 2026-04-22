import random
import sys

import torch
from supconloss import SupConLoss

import config
from data    import load_raw_data, build_class_list, make_loader
from model   import SupConNet
from trainer import train_one_epoch
from utils   import adjust_learning_rate, save_model


# ============================================================================
# HELPERS
# ============================================================================

def _build_model(max_packets: int) -> SupConNet:
    model = SupConNet(
        max_packets    = max_packets,
        hidden_size    = config.HIDDEN_SIZE,
        embedding_size = config.EMBEDDING_SIZE,
        head           = 'mlp',
    )
    if torch.cuda.is_available():
        model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"[Model] {n_params:,} parameters | device: {device}")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main() -> bool:
    # 1. Load data
    X, y, label_names, max_packets = load_raw_data(
        config.RAW_DATA_PATH,
        config.LABEL_MAPPING_PATH,
    )

    # 2. Build class list
    data_list, _ = build_class_list(X, y, config.SAMPLES_PER_CLASS)
    if len(data_list) < 2:
        print("[ERROR] Need at least 2 classes.")
        return False

    random.shuffle(data_list)

    # 3. DataLoader
    train_loader = make_loader(data_list, config.BATCH_SIZE)
    if len(train_loader) == 0:
        print("[ERROR] DataLoader is empty.")
        return False

    # 4. Model, loss, optimiser
    model     = _build_model(max_packets)
    criterion = SupConLoss(temperature=config.TEMPERATURE)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        momentum     = config.MOMENTUM,
        weight_decay = config.WEIGHT_DECAY,
    )

    schedule = "cosine" if config.COSINE_SCHEDULE else "step"
    print(f"[Train] epochs={config.EPOCHS}, bsz={train_loader.batch_size}, "
          f"lr={config.LEARNING_RATE}, temp={config.TEMPERATURE}, lr_schedule={schedule}")

    # 5. Training loop
    for epoch in range(1, config.EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch)

        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}/{config.EPOCHS}  loss={loss:.4f}")

    # 6. Save checkpoint
    save_model(model, optimizer, config.EPOCHS, config.MODEL_PATH)
    print(f"[Done] Model saved to {config.MODEL_PATH}")
    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)
