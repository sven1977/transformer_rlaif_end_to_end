from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<sep>", "<eos>"]


def load_base_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            word, token_id = line.rstrip("\n").split("\t")
            vocab[word] = int(token_id)
    return vocab


def build_token_maps(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    base_vocab = load_base_vocab(path)
    next_id = max(base_vocab.values()) + 1
    for token in SPECIAL_TOKENS:
        if token not in base_vocab:
            base_vocab[token] = next_id
            next_id += 1
    id_to_token = {idx: tok for tok, idx in base_vocab.items()}
    return base_vocab, id_to_token


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    text: list[str]

    def __len__(self):
        return len(self.input_ids)

    def __repr__(self):
        return " ".join(self.text)


class TranslationDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def build_examples(
    data_path: Path,
    token_to_id: dict[str, int],
) -> list[Example]:
    bos = token_to_id["<bos>"]
    sep = token_to_id["<sep>"]
    eos = token_to_id["<eos>"]

    examples: list[Example] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad training row at line {line_no}: expected 2 tab-separated fields.")
            de_sent, en_sent = parts
            # Rows are stored as: german<TAB>english.
            # Train in EN->DE direction: English source, German target.
            src_words = en_sent.split()
            tgt_words = de_sent.split()

            try:
                src_ids = [token_to_id[w] for w in src_words]
                tgt_ids = [token_to_id[w] for w in tgt_words]
            except KeyError as exc:
                missing = str(exc)
                raise ValueError(f"Unknown token {missing} on line {line_no}.") from exc

            full = [bos] + src_ids + [sep] + tgt_ids + [eos]
            full_txt = ["<bos>"] + src_words + ["<sep>"] + tgt_words + ["<eos>"]
            input_ids = full[:-1]
            target_ids = full[1:]

            # Only compute loss on target side (+ eos), not source prompt.
            # Positions up to and including <sep> are ignored.
            sep_pos_in_input = input_ids.index(sep)
            labels = []
            for i, tgt in enumerate(target_ids):
                if i <= sep_pos_in_input:
                    labels.append(-100)
                else:
                    labels.append(tgt)

            examples.append(
                Example(input_ids=input_ids, labels=labels, text=full_txt)
            )

    return examples


def split_train_val(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    idx = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [examples[i] for i in idx if i not in val_idx]
    val = [examples[i] for i in idx if i in val_idx]
    return train, val
