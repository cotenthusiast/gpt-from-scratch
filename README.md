# gpt-from-scratch

A learning-oriented, decoder-only GPT/Transformer implementation in PyTorch,
originally based on Andrej Karpathy's
["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).
It's kept intentionally simple and close to the tutorial structure — a
character-level model trained on Tiny Shakespeare.

## What I implemented independently

**EX1** from the video: replacing the separate per-head `Head` modules with
vectorized multi-head attention (`model.py`). Heads are represented as a
tensor dimension and processed together, rather than as separate Python
module instances.

## Scope

This is not the production/model-release project, and there's no claim of
novelty — it exists to preserve the learning implementation and demonstrate
understanding of the architecture.

## Status

- Tutorial implementation: complete
- EX1 (vectorized multi-head attention): complete
- Further exercises from the video may be revisited later

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (Karpathy's original tutorial hyperparameters)
python train.py

# Generate from a trained checkpoint
python generate.py --checkpoint checkpoints/shakespeare_gpt.pt --tokens 10000
```

`config.py` holds the tutorial hyperparameters; `config_a100.py` (via
`python train.py --a100`) is a larger config used for a Kelvin2 A100 run.
