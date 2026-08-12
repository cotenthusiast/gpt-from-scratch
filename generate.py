"""Load a trained checkpoint and generate text from it."""

import argparse
from pathlib import Path

import torch

from config import GPTConfig
from data import make_encoder_decoder
from model import GPTLanguageModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    chars = checkpoint["chars"]

    model = GPTLanguageModel(len(chars), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, chars, config


def generate(model, chars, tokens, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    _, decode = make_encoder_decoder(chars)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=tokens)[0].tolist()
    return decode(out)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/shakespeare_gpt.pt")
    parser.add_argument("--tokens", type=int, default=5000)
    parser.add_argument("--output", default="outputs/generated.txt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, chars, config = load_model(args.checkpoint, device)

    print(f"Loaded '{config.name}' checkpoint from {args.checkpoint} onto {device}")
    text = generate(model, chars, args.tokens, device, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} characters to {output_path}")

    preview_len = 500
    print("\n--- preview ---")
    print(text[:preview_len])


if __name__ == "__main__":
    main()
