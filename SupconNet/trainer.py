import time

import torch

from utils import AverageMeter, warmup_learning_rate


# ============================================================================
# SINGLE-EPOCH TRAINING
# ============================================================================

def train_one_epoch(loader, model, criterion, optimizer, epoch):
    """
    Run one full pass over *loader*.

    Parameters
    ----------
    loader    : DataLoader yielding ([view1, view2], labels)
    model     : SupConNet
    criterion : SupConLoss
    optimizer : torch.optim.Optimizer
    epoch     : int  (1-indexed)

    Returns
    -------
    float  – average loss for the epoch
    """
    model.train()

    losses     = AverageMeter()
    batch_time = AverageMeter()
    end        = time.time()

    for idx, (views, labels) in enumerate(loader):
        # views is [view1, view2], each (B, max_packets, 3)
        images = torch.cat(views, dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # adjust lr during warmup
        warmup_learning_rate(optimizer, epoch, idx, len(loader))

        # forward
        features = model(images)
        f1, f2   = torch.split(features, [bsz, bsz], dim=0)
        features = torch.stack([f1, f2], dim=1)   # (B, 2, embed_dim)

        loss = criterion(features, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  NaN/Inf loss at batch {idx} – skipping")
            continue

        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg