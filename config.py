"""Model/training hyperparameter presets."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    name: str
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    n_embed: int
    n_head: int
    n_layer: int
    dropout: float


# Original tutorial configuration (Karpathy's final settings), kept as the
# small/local default so the tutorial result stays easy to reproduce.
TUTORIAL = GPTConfig(
    name="tutorial",
    batch_size=64,
    block_size=256,
    max_iters=5000,
    eval_interval=300,
    eval_iters=200,
    learning_rate=3e-4,
    n_embed=384,
    n_head=6,
    n_layer=6,
    dropout=0.2,
)

# Stronger configuration for a single Kelvin2 A100. More context, capacity,
# and iterations than the tutorial config, but still small relative to what
# Tiny Shakespeare (~1.1M characters) can support without gross overfitting.
A100 = GPTConfig(
    name="a100",
    batch_size=96,
    block_size=512,
    max_iters=6000,
    eval_interval=500,
    eval_iters=200,
    learning_rate=3e-4,
    n_embed=512,
    n_head=8,
    n_layer=8,
    dropout=0.2,
)

PRESETS = {"tutorial": TUTORIAL, "a100": A100}
