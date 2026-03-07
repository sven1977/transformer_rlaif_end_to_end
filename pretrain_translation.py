import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer import MicroTransformer

from utils import (
    build_examples,
    build_token_maps,
    evaluate,
    make_collate_fn,
    save_checkpoint,
    split_train_val,
    train_epoch,
    TranslationDataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MicroTransformer on dummy EN->DE translation pairs.")
    p.add_argument("--vocab-path", type=Path, default=Path("dummy_vocab.tsv"))
    p.add_argument("--data-path", type=Path, default=Path("dummy_train.tsv"))
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-k", type=int, default=16)
    p.add_argument("--d-v", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    token_to_id, _ = build_token_maps(args.vocab_path)
    examples = build_examples(args.data_path, token_to_id)
    train_examples, val_examples = split_train_val(examples, args.val_ratio, args.seed)

    max_seq_len = max(len(ex) for ex in examples)
    vocab_size = max(token_to_id.values()) + 1
    pad_id = token_to_id["<pad>"]

    train_ds = TranslationDataset(train_examples)
    val_ds = TranslationDataset(val_examples)
    collate_fn = make_collate_fn(pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MicroTransformer(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        num_transformer_blocks=args.num_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_k": args.d_k,
        "d_v": args.d_v,
        "d_ff": args.d_ff,
        "num_blocks": args.num_blocks,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    best_val = math.inf
    epochs_without_improve = 0

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"vocab={vocab_size} max_seq_len={max_seq_len}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_loss = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)

        print(f"epoch={epoch:03d} train_loss={train_loss:.8f} val_loss={val_loss:.8f}")

        if epoch % args.checkpoint_every == 0:
            ckpt_path = args.checkpoints_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            save_checkpoint(
                args.checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    "early_stop=True "
                    f"(no val improvement for {args.patience} consecutive epochs)"
                )
                break

    save_checkpoint(
        args.checkpoints_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
    )


if __name__ == "__main__":
    main()
