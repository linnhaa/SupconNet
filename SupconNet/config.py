# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
RAW_DATA_PATH      = 'data_pretrain/50labels500_mac_2cdns/embedding_data/raw_packet_data.csv'
LABEL_MAPPING_PATH = 'data_pretrain/50labels500_mac_2cdns/embedding_data/label_mapping.csv'
MODEL_PATH         = 'model/supconnet.pth'

# Training
BATCH_SIZE    = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY  = 1e-4
MOMENTUM      = 0.9
EPOCHS        = 1
TEMPERATURE   = 0.1

# Learning-rate schedule
LR_DECAY_EPOCHS = [700, 800, 900]
LR_DECAY_RATE   = 0.1
COSINE_SCHEDULE = True
WARMUP          = True
WARMUP_EPOCHS   = 10
WARMUP_FROM     = 0.01

# Model architecture
HIDDEN_SIZE    = 256
EMBEDDING_SIZE = 128

# Data processing
MIN_PACKETS      = 100
MAX_PACKETS      = 25000
SAMPLES_PER_CLASS = 250