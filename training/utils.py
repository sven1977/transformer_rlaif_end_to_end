from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.utils import Example


def make_collate_fn(pad_id: int):
    def collate(batch: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in batch)
        bsz = len(batch)
        x = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, ex in enumerate(batch):
            L = len(ex.input_ids)
            x[i, :L] = torch.tensor(ex.input_ids, dtype=torch.long)
            y[i, :L] = torch.tensor(ex.labels, dtype=torch.long)
        return x, y

    return collate


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(payload, path)
