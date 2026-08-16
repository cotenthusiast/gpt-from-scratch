import argparse

import torch

from data import get_vocab
from model import BigramLanguageModel


def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained checkpoint.')
    parser.add_argument('--checkpoint', default='checkpoints/shakespeare_gpt.pt')
    parser.add_argument('--tokens', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device)

    chars = checkpoint['chars']
    encode, decode = get_vocab(chars)

    model = BigramLanguageModel(
        checkpoint['vocab_size'],
        checkpoint['n_embed'],
        checkpoint['block_size'],
        checkpoint['n_head'],
        checkpoint['n_layer'],
        checkpoint['dropout'],
    )
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=args.tokens)[0].tolist()))


if __name__ == '__main__':
    main()
