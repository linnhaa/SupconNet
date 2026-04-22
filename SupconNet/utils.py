import math
import os

import numpy as np
import torch

import config


# ============================================================================
# AVERAGE METER
# ============================================================================

class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


# ============================================================================
# LEARNING-RATE HELPERS
# ============================================================================

def _cosine_lr(base_lr, epoch):
    eta_min = base_lr * (config.LR_DECAY_RATE ** 3)
    return eta_min + (base_lr - eta_min) * (
        1 + math.cos(math.pi * epoch / config.EPOCHS)
    ) / 2


def adjust_learning_rate(optimizer, epoch):
    """Decay LR following cosine or step schedule."""
    if config.COSINE_SCHEDULE:
        lr = _cosine_lr(config.LEARNING_RATE, epoch)
    else:
        steps = np.sum(epoch > np.asarray(config.LR_DECAY_EPOCHS))
        lr = config.LEARNING_RATE * (config.LR_DECAY_RATE ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_learning_rate(optimizer, epoch, batch_id, total_batches):
    """Linear warmup during the first WARMUP_EPOCHS epochs."""
    if not config.WARMUP or epoch > config.WARMUP_EPOCHS:
        return

    warmup_to = (
        _cosine_lr(config.LEARNING_RATE, config.WARMUP_EPOCHS)
        if config.COSINE_SCHEDULE
        else config.LEARNING_RATE
    )

    p = (batch_id + (epoch - 1) * total_batches) / (
        config.WARMUP_EPOCHS * total_batches
    )
    lr = config.WARMUP_FROM + p * (warmup_to - config.WARMUP_FROM)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_model(model, optimizer, epoch, save_path):
    """Save model checkpoint to *save_path*."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'max_packets':    getattr(model, 'max_packets', config.MAX_PACKETS),
                'hidden_size':    config.HIDDEN_SIZE,
                'embedding_size': config.EMBEDDING_SIZE,
            },
        },
        save_path,
    )