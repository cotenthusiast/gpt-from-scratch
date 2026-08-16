import argparse

import torch

from data import load_data, get_batch
from model import BigramLanguageModel

torch.manual_seed(1337)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, train_data, val_data, cfg.block_size, cfg.batch_size, cfg.device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description='Train the GPT on Tiny Shakespeare.')
    parser.add_argument('--a100', action='store_true', help='use the Kelvin2 A100 config instead of the tutorial config')
    parser.add_argument('--checkpoint', default='checkpoints/shakespeare_gpt.pt')
    args = parser.parse_args()

    if args.a100:
        import config_a100 as cfg
    else:
        import config as cfg

    train_data, val_data, chars, vocab_size, encode, decode = load_data()

    model = BigramLanguageModel(vocab_size, cfg.n_embed, cfg.block_size, cfg.n_head, cfg.n_layer, cfg.dropout)
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"Training on {cfg.device}")

    for iter in range(cfg.max_iters):
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, cfg.block_size, cfg.batch_size, cfg.device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': cfg.max_iters,
        'vocab_size': vocab_size,
        'chars': chars,
        'n_embed': cfg.n_embed,
        'block_size': cfg.block_size,
        'n_head': cfg.n_head,
        'n_layer': cfg.n_layer,
        'dropout': cfg.dropout,
    }, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == '__main__':
    main()
