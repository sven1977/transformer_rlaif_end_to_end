# Transformer + RLAIF, end-to-end, everything in one

This repository is a sandbox for building an end-to-end LLM pre-training system from scratch.
We then extend the pipeline into post-training with reinforcement learning from AI feedback (RLAIF).
Finally, the trained model can be inference-served through a web interface.

The primary purpose of this repo is for users to have a hands-on, end-to-end, tutorial-style example
for how such LLM pre- and post-training pipelines can be implemented and trained, at any scale, using Ray.

Also, one nice side effect of this endeavor was for myself (@sven1977) to learn the core concepts and challenges
of RL in connection with LLMs, from first-principles and with only minor help from AIs :)
... Most of the code was initially written by myself, then refined and double-checked through Codex and/or
Claude.


## Target workflow

1. Build a custom decoder-only transformer in PyTorch, based on the architecture introduced in the 2017 paper *Attention Is All You Need*.
1. Generate a dummy English-to-German translation dataset with train/validation splits. The dataset is intentionally synthetic and contains
   only simple source sentences such as `the yellow fox ran around the pink garden`, with German labels:
   `der gelbe Fuchs lief um den pinken Garten`.
1. Pre-train the transformer from scratch on that dataset using the validation set as stop indicator.
1. Training can run in distributed fashion by leveraging:
   - data parallelism
   - tensor parallelism
   - pipeline parallelism 
   The same code path can thus scale toward larger production-style setups.
1. Use Ray Train to orchestrate the various distributed training workers and define the training and evaluation loop.
1. Produce checkpoints along the way.
1. Implement PPO for LLMs with KL-reward modifier from scratch, as a fun exercise.
1. Extend the pretrained model with RLAIF post-training so translations remain faithful to the source
   while also becoming more semantically sensible. A fixed LLM-based judge provides rewards
1. After pre- and post-training, add inference and a small web interface for serving the final model.

## Project intent

The broader goal is not just to train a toy translator, but to document the full lifecycle of a modern model project in a compact, understandable codebase:

- model implementation from first principles
- synthetic data generation
- distributed training
- checkpointing and evaluation
- preference-style post-training with an LLM judge
- inference and serving

The synthetic dataset is useful because it keeps iteration cheap while still letting us exercise the mechanics of tokenization, batching, sequence modeling, distributed training, checkpointing, and later reward modeling or preference optimization.

## Current repo structure

- [`models/micro_transformer.py`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/models/micro_transformer.py): custom decoder-only transformer implementation in PyTorch, including causal attention and tensor-parallel-aware layers.
- [`data/generate_dummy_pretrain_data.py`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/data/generate_dummy_pretrain_data.py): generates a synthetic German-to-English translation dataset and vocabulary file.
- [`training/pretrain_translation.py`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/training/pretrain_translation.py): Ray-based pretraining entry point with checkpoint writing and distributed worker setup.
- [`training/utils.py`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/training/utils.py): training-side dataset, token, batching, and split helpers.
- [`evaluation/utils.py`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/evaluation/utils.py): evaluation helpers.
- [`data/dummy_train.tsv`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/data/dummy_train.tsv): generated translation pairs.
- [`data/dummy_vocab.tsv`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/data/dummy_vocab.tsv): generated vocabulary.
- [`.checkpoints/`](/Users/marlenesven/Library/CloudStorage/Dropbox/Projects/transformer_rlaif_end_to_end/.checkpoints): saved training checkpoints.

## Distributed training direction

The intended training story is to make the model runnable on anything from a local machine to a larger cluster:

- Data parallelism replicates model shards across workers and averages gradients.
- Tensor parallelism splits compatible parts of the transformer across workers.
- Pipeline parallelism is part of the target architecture so deeper or wider production models can be split stage-by-stage across devices.
- Ray provides the training orchestration layer and checkpoint lifecycle.

At the moment, the codebase already sketches the data-parallel plus tensor-parallel pieces. Pipeline parallelism is still part of the intended roadmap rather than a completed implementation.

## Planned RLAIF phase

After pretraining, the plan is to run RLAIF post-training using a fixed LLM as judge.

The key idea is that the synthetic source sentences are sometimes grammatically valid but semantically odd. For example, a sentence may describe a `yellow fox`, which is close to the source distribution but not especially natural. In that phase, the judge should reward outputs that:

- stay as close to the source meaning as possible
- remain fluent and complete
- prefer sensible translations when the source contains unnatural combinations

In practice, that means encouraging outputs closer to `the red fox ...` than `the yellow fox ...` when doing so improves semantic plausibility without breaking the underlying intent of the sentence.

## Roadmap

- [x] Build a minimal custom decoder-only transformer in PyTorch
- [x] Generate a synthetic DE->EN translation dataset
- [x] Add Ray-based pretraining with checkpoint emission
- [ ] Complete pipeline parallel support
- [ ] Add a fixed-judge RLAIF post-training loop
- [ ] Implement PPO from scratch
- [ ] Add inference code for the post-trained model
- [ ] Serve the model behind a web interface

## Notes

- This repository is intentionally educational and experimental.
- Some components are already implemented, while others are explicitly placeholders for the next stages.
- The code is meant to make the full workflow understandable end to end, not just to maximize model quality on this toy dataset.
