import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
from ray import train
from ray.train.torch import get_device
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.utils import TranslationDataset, build_examples, build_token_maps, split_train_val
from models.micro_transformer import MicroTransformer
from training.utils import make_collate_fn


def ray_train_pretraining_loop(config: dict) -> None:
    ctx = train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    ddp_size = config["ddp"]
    tp_size = config["tp"]
    assert tp_size * ddp_size == world_size
    assert world_size % tp_size == 0
    # Global rank0 worker is responsible for printing stats and checkpointing.
    is_rank0 = world_rank == 0

    # Rank layout:
    # 8 workers, ddp=2, tp=4: ranks 0 to 7.

    # Each worker:
    # - has to figure out, where the other weights shards are so it can reduce
    #   with them during model forward passes. These need to be those shards that
    #   operated on the same batch slice (DDP group matches)
    #   -> each worker has 3 other "tp_peers" with the same dp-rank
    ddp_rank = world_rank // tp_size  # 0 or 1 in our example
    # - has to figure out, where the other DDP shards with the same(!) weights shards
    #   are so it can average gradients with them.
    #   -> each worker has 1 other "ddp_peer" with the same tp-rank
    tp_rank = world_rank % tp_size  # 0, 1, 2, 3 in our example

    ddp_peer_groups = []
    for t in range(tp_size):
        ddp_peer_groups.append(dist.new_group([i * tp_size + t for i in range(ddp_size)]))
    ddp_peer_group = ddp_peer_groups[tp_rank]

    tp_peer_groups = []
    for d in range(ddp_size):
        tp_peer_groups.append(dist.new_group([d * tp_size + i for i in range(tp_size)]))
    tp_peer_group = tp_peer_groups[ddp_rank]

    # Seed this train worker.
    random.seed(config["seed"] + world_rank)
    torch.manual_seed(config["seed"] + world_rank)

    # Read in complete dataset. Through random shuffling, we make sure that each trainer worker
    # - most of the time - reads in unique information for its own mini/microbatch.
    token_to_id, _ = build_token_maps(Path(config["vocab_path"]))
    examples = build_examples(Path(config["data_path"]), token_to_id)
    train_examples, val_examples = split_train_val(examples, config["validation_ratio"], config["seed"])

    max_seq_len = max(len(ex) for ex in examples)
    vocab_size = max(token_to_id.values()) + 1
    pad_id = token_to_id["<pad>"]

    train_ds = TranslationDataset(train_examples)
    val_ds = TranslationDataset(val_examples)
    collate_fn = make_collate_fn(pad_id)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=ddp_size,
        rank=ddp_rank,
        shuffle=True,
        seed=config["seed"],
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=ddp_size,
        rank=ddp_rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    if torch.cuda.is_available():
        device = get_device()
        torch.cuda.set_device(device)
    else:
        # Keep CPU execution on Apple Silicon when MPS is present:
        # current PyTorch FSDP paths may call torch.mps.current_device(), which
        # is missing in some MPS builds and crashes during DDP initialization.
        device = torch.device("cpu")

    base_model = MicroTransformer(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_k=config["d_k"],
        d_v=config["d_v"],
        d_ff=config["d_ff"],
        num_transformer_blocks=config["num_blocks"],
        tp_group=tp_peer_group,
    ).to(device)
    model = DDP(base_model, process_group=ddp_peer_group)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = math.inf
    epochs_without_improve = 0

    if is_rank0:
        print(
            f"world_size={world_size} train={len(train_ds)} val={len(val_ds)} "
            f"vocab={vocab_size} max_seq_len={max_seq_len} device={device}"
        )

    for epoch in range(1, config["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            train_total += loss.item()
            train_count += 1

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_total += loss.item()
                val_count += 1

        train_loss = _allreduce_avg(train_total, train_count, device)
        val_loss = _allreduce_avg(val_total, val_count, device)

        if is_rank0:
            print(f"epoch={epoch:03d} train_loss={train_loss:.8f} val_loss={val_loss:.8f}")

        save_regular = (epoch % config["checkpoint_every"] == 0)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if is_rank0 and (save_regular or improved):
            ckpt_name = "best.pt" if improved else f"epoch_{epoch:03d}.pt"
            final_path = Path(config["checkpoints_dir"]) / ckpt_name
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
                out_path=final_path,
            )
        train.report(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "early_stop": False,
            },
        )

        if epochs_without_improve >= config["patience"]:
            if is_rank0:
                print(
                    "early_stop=True "
                    f"(no val improvement for {config['patience']} consecutive epochs)"
                )
            break

    if is_rank0:
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            config=config,
            out_path=Path(config["checkpoints_dir"]) / "last.pt",
        )
    train.report(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "early_stop": epochs_without_improve >= config["patience"],
        }
    )
