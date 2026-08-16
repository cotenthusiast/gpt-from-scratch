import torch

# Hyperparameters (Kelvin2 A100 configuration - larger than the tutorial
# defaults for a single A100, still modest for Tiny Shakespeare)
batch_size = 96
block_size = 512
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 512
n_head = 8
n_layer = 8
dropout = 0.2
