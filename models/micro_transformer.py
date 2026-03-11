"""Fully self-contained transformer model using the decoder-only architecture from [1]."""

from math import sin, cos

import torch
import torch.distributed as dist
from torch import nn


class MicroTransformer(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        num_transformer_blocks: int,
        tp_group=None,  # tensor parallel group
    ):
        """Initializes a MicroTransformer instance.

        Args:
              max_seq_len: Maximum sequence length that would fit into the transformer's context window.
              vocab_size: Number of tokens in the used vocabulary. This is the same as the number of rows in the
                embedding matrix.
              d_model: Dimensionality of each token embedding vector. This is also the dimension of each token going
                into and coming out of each transformer block.
              num_heads: Number of heads in the multi-head attention part of each transformer block.
              d_k: Dimensionality of queue- and key vectors of each token.
              d_v: Dimensionality of value vectors of each token.
              d_ff: Dimensionality of the first layer of the positional feed-forward layer after the multi-head
                attention block in each transformer block.
              num_transformer_blocks: Number of transformer blocks.
        """
        super().__init__()

        # The initial embedding layer.
        # Note that the weights of its matrix are shared with the final output layer's (logits) weight matrix, which is
        # the transpose of this embedding matrix mapping feature vectors for each token back to each tokens' logit.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # The positional encoding values.
        self.register_buffer(
            "positional_encodings",
            self._get_positional_encodings(max_seq_len, d_model),
            persistent=True,
        )

        # The n transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            _TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                tp_group=tp_group,
            )
            for _ in range(num_transformer_blocks)
        ])

    def forward(self, x):
        """Forward pass through the transformer model."""
        # Embedding lookup.
        x = self.embedding(x)  # [B, T, d_model]
        T = x.size(1)
        # Add positional encodings (broadcast along batch dim).
        x = x + self.positional_encodings[:T].unsqueeze(0)
        # Push through all transformer blocks.
        for trans in self.transformer_blocks:
            x = trans(x)
        # Final logits layer (use embedding weight matrix' transpose).
        logits = x @ self.embedding.weight.T  # [B, T, vocab_size]

        return logits

    def _get_positional_encodings(self, max_seq_len, d_model):
        encodings = torch.zeros((max_seq_len, d_model), dtype=torch.float32)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                encodings[pos, i] = sin(pos / 10000 ** (2*i / d_model))
                encodings[pos, i+1] = cos(pos / 10000 ** (2*i / d_model))
        return encodings


class _TransformerBlock(nn.Module):
    def __init__(self, *, d_model, num_heads, d_k, d_v, d_ff, tp_group=None):
        super().__init__()

        self.tp_group = tp_group
        if dist.is_available() and dist.is_initialized() and self.tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_size = 1
            self.tp_rank = 0

        # Make sure the tp-split makes sense.
        # We need to split the H heads amongst the TP shards.
        assert num_heads % self.tp_size == 0
        # We also need to column-split the first weights matrix of the positional FF net.
        assert d_ff % self.tp_size == 0

        self._num_local_heads = num_heads // self.tp_size
        #self._num_heads = num_heads
        self._d_k = d_k
        self._d_v = d_v

        # Multi-head attention block.
        # Create the Wq, Wk, Wv attention matrices (column parallel).
        self._Wq = nn.Linear(d_model, self._num_local_heads * d_k, bias=False)
        self._Wk = nn.Linear(d_model, self._num_local_heads * d_k, bias=False)
        self._Wv = nn.Linear(d_model, self._num_local_heads * d_v, bias=False)

        # Create final Wo matrix (row parallel).
        self._Wo = nn.Linear(self._num_local_heads * d_v, d_model, bias=False)

        self._layernorm_1 = nn.LayerNorm(d_model)
        self._layernorm_2 = nn.LayerNorm(d_model)

        # Positional feed-forward block.
        # 1st linear.
        self._positional_ff = nn.Sequential(
            # 1st layer is column parallel.
            nn.Linear(d_model, d_ff // self.tp_size, bias=True),
            # Activate locally (no all-gather/all-reduce).
            nn.ReLU(),
            # 2nd payer is row parallel.
            nn.Linear(d_ff // self.tp_size, d_model),
            # Then we all-reduce sum to get final result for both matmuls combined.
        )

    def forward(self, x):
        in_ = x

        B, T, _ = x.size()

        # Multi-head attention:
        # ---------------------
        Q = self._Wq(x)  # [B, T, H*d_k]
        K = self._Wk(x)  # [B, T, H*d_k]
        V = self._Wv(x)  # [B, T, H*d_v]

        # Separate heads (H) dim from last dim and permute to push H right after B.
        Q = Q.reshape((B, T, self._num_local_heads, self._d_k)).permute((0, 2, 1, 3))  # [B, H, T, d_k]
        V = V.reshape((B, T, self._num_local_heads, self._d_v)).permute((0, 2, 1, 3))  # [B, H, T, d_v]
        # Separate heads (H) dim from last dim and permute to push H right after B
        # AND flip T and last dim for upcoming matmul with Q.
        K_T = K.reshape((B, T, self._num_local_heads, self._d_k)).permute((0, 2, 3, 1))  # [B, H, d_k, T]

        # Attention scores.
        S = torch.matmul(Q, K_T)  # [B, H, T, T]
        # Scale.
        S = S / (self._d_k ** 0.5)  # [B, H, T, T]

        # Apply causal mask.
        mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=Q.dtype, device=Q.device),
            diagonal=1,  # first column (0) is all non-masked
        )
        S_masked = S + mask  # [B, H, T, T]

        # Softmax over rows of S.
        S_softmax = torch.nn.functional.softmax(S_masked, dim=-1)  # [B, H, T, T]

        # Attention values: Multiply with V.
        A = torch.matmul(S_softmax, V)  # [B, H, T, d_v]
        A = A.permute((0, 2, 1, 3)).reshape((B, T, -1))  # [B, T, H*d_v]
        # Apply one last linear projection to get back to d_model.
        O = self._Wo(A)  # [B, T, d_model]
        # All-reduce sum over the TP row-parallel final Wo shards to get global results.
        if self.tp_group is not None and self.tp_size > 1:
            dist.all_reduce(O, op=dist.ReduceOp.SUM, group=self.tp_group)

        # Add (residual) and norm.
        O_res = in_ + O
        in_ = self._layernorm_1(O_res)

        # Positional feed forward:
        # ------------------------
        O = self._positional_ff(in_)
        # All-reduce sum over the TP row-parallel 2nd layer shards to get global results.
        if self.tp_group is not None and self.tp_size > 1:
            dist.all_reduce(O, op=dist.ReduceOp.SUM, group=self.tp_group)

        # Add (residual) and norm.
        O_res = in_ + O
        final_out = self._layernorm_2(O_res)

        return final_out


if __name__ == "__main__":
    torch.manual_seed(0)

    model = MicroTransformer(
        max_seq_len=8,
        vocab_size=32,
        d_model=16,
        num_heads=2,
        d_k=8,
        d_v=8,
        d_ff=32,
        num_transformer_blocks=2,
    )

    # Dummy token IDs in [0, vocab_size).
    dummy_batch = torch.randint(low=0, high=32, size=(2, 8), dtype=torch.long)

    print("dummy_batch shape:", dummy_batch.shape)
    out = model(dummy_batch)
    print("output shape:", out.shape)
