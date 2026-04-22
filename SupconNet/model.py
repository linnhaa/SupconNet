import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SFE(nn.Module):
    """
    DF-style 1-D CNN encoder for raw packet sequences.

    Input  : (B, max_packets, 3)   – [time, direction, size]
    Output : (B, hidden_size)
    """

    _KERNEL  = 8
    _POOL_K  = 8
    _POOL_S  = 4

    def __init__(self, max_packets: int, hidden_size: int = 256):
        super().__init__()
        self.max_packets = max_packets
        self.hidden_size = hidden_size

        K, PK, PS = self._KERNEL, self._POOL_K, self._POOL_S

        # --- conv blocks ---
        self.conv1   = nn.Conv1d(  3,  32, K)
        self.conv1_1 = nn.Conv1d( 32,  32, K)
        self.conv2   = nn.Conv1d( 32,  64, K)
        self.conv2_2 = nn.Conv1d( 64,  64, K)
        self.conv3   = nn.Conv1d( 64, 128, K)
        self.conv3_3 = nn.Conv1d(128, 128, K)
        self.conv4   = nn.Conv1d(128, 256, K)
        self.conv4_4 = nn.Conv1d(256, 256, K)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool1 = nn.MaxPool1d(PK, PS)
        self.pool2 = nn.MaxPool1d(PK, PS)
        self.pool3 = nn.MaxPool1d(PK, PS)
        self.pool4 = nn.MaxPool1d(PK, PS)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.drop4 = nn.Dropout(0.1)

        # infer FC input dim via a dummy forward pass
        with torch.no_grad():
            flat_dim = self._conv_forward(torch.zeros(1, 3, max_packets)).shape[1]

        self.fc = nn.Linear(flat_dim, hidden_size)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pad(x):
        """Same-length padding for kernel_size=8 (pad 3 left, 4 right)."""
        return F.pad(x, (3, 4))

    def _conv_forward(self, x):
        """Pure conv/pool pipeline – reused by dummy pass and real forward."""
        p = self._pad

        # Block 1 – ELU
        x = F.elu(self.conv1(p(x)))
        x = F.elu(self.bn1(self.conv1_1(p(x))))
        x = self.drop1(self.pool1(p(x)))

        # Block 2 – ReLU
        x = F.relu(self.conv2(p(x)))
        x = F.relu(self.bn2(self.conv2_2(p(x))))
        x = self.drop2(self.pool2(p(x)))

        # Block 3 – ReLU
        x = F.relu(self.conv3(p(x)))
        x = F.relu(self.bn3(self.conv3_3(p(x))))
        x = self.drop3(self.pool3(p(x)))

        # Block 4 – ReLU
        x = F.relu(self.conv4(p(x)))
        x = F.relu(self.bn4(self.conv4_4(p(x))))
        x = self.drop4(self.pool4(p(x)))

        return x.view(x.size(0), -1)   # flatten

    def forward(self, x):
        # x: (B, max_packets, 3)
        x = x.transpose(1, 2)          # → (B, 3, max_packets)
        return self.fc(self._conv_forward(x))


# ============================================================================
# FULL NETWORK  (encoder + projection head)
# ============================================================================

class SupConNet(nn.Module):
    """
    Backbone + MLP projection head for Supervised Contrastive Learning.

    Output is L2-normalised – ready to feed directly into SupConLoss.
    """

    def __init__(
        self,
        max_packets:    int,
        hidden_size:    int = config.HIDDEN_SIZE,
        embedding_size: int = config.EMBEDDING_SIZE,
        head:           str = 'mlp',
    ):
        super().__init__()
        self.max_packets = max_packets

        self.encoder = SFE(max_packets, hidden_size)

        if head == 'linear':
            self.head = nn.Linear(hidden_size, embedding_size)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, embedding_size),
            )
        else:
            raise ValueError(f"Unknown head type: '{head}'")

    def forward(self, x):
        return F.normalize(self.head(self.encoder(x)), dim=1)