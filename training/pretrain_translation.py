import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch import nn

from training.ray_train_pretraining_loop import ray_train_pretraining_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MicroTransformer on dummy EN->DE translation pairs with Ray + DDP.")
    p.add_argument("--vocab-path", type=Path, default=Path("data/dummy_vocab.tsv").absolute())
    p.add_argument("--data-path", type=Path, default=Path("data/dummy_train.tsv").absolute())
    p.add_argument("--checkpoints-dir", type=Path, default=Path(".checkpoints").absolute())
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128, help="The model's main dimension. Each token goes into a transformer block and comes out of it with this dim. This is also the embedding dim.")
    p.add_argument("--num-heads", type=int, default=8, help="The number of attention heads per transformer block.")
    p.add_argument("--d-k", type=int, default=16, help="Dimension of the queue- and key-tensors per token.")
    p.add_argument("--d-v", type=int, default=16, help="Dimension of the value-tensors per token.")
    p.add_argument("--d-ff", type=int, default=256, help="Dimension of the positional feed-forward NN expansion layer.")
    p.add_argument("--num-blocks", type=int, default=4, help="Number of transformer blocks.")
    p.add_argument("--ddp", type=int, default=2, help="Number of distr. data parallelism (DDP) shards.")
    p.add_argument("--tp", type=int, default=2, help="Number of tensor parallelism (TP) shards.")
    p.add_argument("--pp", type=int, default=1, help="Number of pipeline parallelism (PP) shards.")
    p.add_argument("--name", type=str, default="pretrain_en_to_de")
    return p.parse_args()


def _allreduce_avg(total: float, count: int, device: torch.device) -> float:
    t = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1].clamp(min=1.0)).item()


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
    out_path: Path,
) -> None:
    model_state = model.state_dict()

    payload = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def main() -> None:
    args = parse_args()
    if args.ddp < 1:
        raise ValueError("--ddp must be >= 1")
    if args.tp < 1:
        raise ValueError("--tp must be >= 1")
    if args.pp < 1:
        raise ValueError("--pp must be >= 1")

    config = {
        "vocab_path": str(args.vocab_path),
        "data_path": str(args.data_path),
        "checkpoints_dir": str(args.checkpoints_dir),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "patience": args.patience,
        "checkpoint_every": args.checkpoint_every,
        "grad_clip": args.grad_clip,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_k": args.d_k,
        "d_v": args.d_v,
        "d_ff": args.d_ff,
        "num_blocks": args.num_blocks,
        "ddp": args.ddp,
        "tp": args.tp,
        "pp": args.pp,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=ray_train_pretraining_loop,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=args.ddp * args.tp, use_gpu=torch.cuda.is_available()),
        run_config=RunConfig(name=args.name),
    )
    result = trainer.fit()
    metrics = result.metrics
    print(
        "training_complete "
        f"epoch={metrics.get('epoch')} "
        f"train_loss={metrics.get('train_loss')} "
        f"val_loss={metrics.get('val_loss')} "
        f"best_val_loss={metrics.get('best_val_loss')}"
    )


if __name__ == "__main__":
    main()
