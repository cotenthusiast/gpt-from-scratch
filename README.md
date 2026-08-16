# gpt-from-scratch

A character-level GPT trained on Tiny Shakespeare, following Andrej Karpathy's
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).

The tutorial implementation (as three standalone scripts: `bigram.py`,
`train.py`, `v2.py`) is preserved at the `tutorial-complete` git tag. This
version of the repo is a light refactor of that same model into separate
modules, plus a training pipeline for a Kelvin2 A100.

## Structure

- `model.py` — the Transformer: attention head, multi-head attention,
  feed-forward network, pre-LN block, `BigramLanguageModel`
- `data.py` — character-level tokenizer and batching
- `config.py` — hyperparameters (Karpathy's tutorial settings)
- `config_a100.py` — larger hyperparameters for a single Kelvin2 A100
- `train.py` — training loop, periodic train/val loss estimation, checkpoint saving
- `generate.py` — loads a checkpoint and generates text
- `input.txt` — Tiny Shakespeare
- `scripts/train_kelvin2.slurm` — Kelvin2 SLURM job (1x A100)
- `checkpoints/`, `outputs/`, `logs/` — local/HPC run artifacts (gitignored)

## Local usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Small/local config (Karpathy's original tutorial hyperparameters)
python train.py

# Generate from a trained checkpoint
python generate.py --checkpoint checkpoints/shakespeare_gpt.pt --tokens 10000
```

## Configs

| | `config.py` | `config_a100.py` |
|---|---|---|
| batch_size | 64 | 96 |
| block_size | 256 | 512 |
| n_embed | 384 | 512 |
| n_head | 6 | 8 |
| n_layer | 6 | 8 |
| dropout | 0.2 | 0.2 |
| learning_rate | 3e-4 | 3e-4 |
| max_iters | 5000 | 6000 |

`config.py` reproduces Karpathy's final tutorial settings and is the sane
default for a laptop/CPU run (`python train.py`). `config_a100.py` (used via
`python train.py --a100`) is a deliberately modest step up in context and
capacity for a single Kelvin2 A100 — Tiny Shakespeare is only ~1.1M
characters, so this stops well short of a size that would just overfit or
waste GPU time.

## Kelvin2

```bash
ssh kelvin2
cd /mnt/scratch2/users/$USER
git clone https://github.com/cotenthusiast/gpt-from-scratch repos/gpt-from-scratch

# The login node's bare `python3` is 3.6 and `module` isn't available over
# plain SSH; use the 3.10 interpreter directly so the venv is unambiguous.
/opt/apps/python3/3.10.5/gcc-9.3.0/bin/python3 -m venv venvs/gpt-from-scratch
source venvs/gpt-from-scratch/bin/activate
pip install -r repos/gpt-from-scratch/requirements.txt

cd repos/gpt-from-scratch
sbatch scripts/train_kelvin2.slurm
squeue -u $USER
```
