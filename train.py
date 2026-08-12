"""Training loop for the char-level GPT on Tiny Shakespeare."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from config import PRESETS
from data import get_batch, load_data
from model import GPTLanguageModel


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    out = {}
    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config.block_size, config.batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def train(config, device, data_path, checkpoint_path, seed=1337):
    torch.manual_seed(seed)

    train_data, val_data, chars, vocab_size = load_data(data_path)

    model = GPTLanguageModel(vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters")
    print(f"Training on {device} with preset '{config.name}'")

    losses = {"train": float("nan"), "val": float("nan")}
    start = time.time()
    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config, device)
            elapsed = time.time() - start
            print(
                f"step {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f} ({elapsed:.0f}s)"
            )

        xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": config.max_iters,
            "config": asdict(config),
            "chars": chars,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    return model, chars


def main():
    parser = argparse.ArgumentParser(description="Train a char-level GPT on Tiny Shakespeare.")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="tutorial")
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--checkpoint", default="checkpoints/shakespeare_gpt.pt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PRESETS[args.preset]

    train(config, device, args.data, args.checkpoint, seed=args.seed)


if __name__ == "__main__":
    main()
