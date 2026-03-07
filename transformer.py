from math import sin, cos

import torch
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
    ):
        super().__init__()

        # The initial embedding layer.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # The positional encoding values.
        pe = self._get_positional_encodings(max_seq_len, d_model)
        self.register_buffer("positional_encodings", pe, persistent=True)

        # The n transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            _TransformerBlock(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, d_ff=d_ff)
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
        # Final logits layer.
        logits = x @ self.embedding.weight.T  # [B, T, vocab_size]
        ## Softmax.
        #probs = torch.nn.functional.softmax(logits, dim=-1)
        return logits

    def _get_positional_encodings(self, max_seq_len, d_model):
        encodings = torch.zeros((max_seq_len, d_model), dtype=torch.float32)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                encodings[pos, i] = sin(pos / 10000 ** (2*i / d_model))
                encodings[pos, i+1] = cos(pos / 10000 ** (2*i / d_model))
        return encodings


class _TransformerBlock(nn.Module):
    def __init__(self, *, d_model, num_heads, d_k, d_v, d_ff):
        super().__init__()

        self._num_heads = num_heads
        self._d_k = d_k

        # Multi-head attention block.
        # Create the Wq, Wk, Wv attention matrices.
        self._Wq = nn.Linear(d_model, num_heads * d_k, bias=False)
        self._Wk = nn.Linear(d_model, num_heads * d_k, bias=False)
        self._Wv = nn.Linear(d_model, num_heads * d_v, bias=False)

        self._Wo = nn.Linear(num_heads * d_v, d_model, bias=False)

        self._layernorm_1 = nn.LayerNorm(d_model)
        self._layernorm_2 = nn.LayerNorm(d_model)

        # Positional feed-forward block.
        # 1st linear.
        self._positional_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
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
        Q = Q.reshape((B, T, self._num_heads, -1)).permute((0, 2, 1, 3))  # [B, H, T, d_k]
        V = V.reshape((B, T, self._num_heads, -1)).permute((0, 2, 1, 3))  # [B, H, T, d_v]
        # Separate heads (H) dim from last dim and permute to push H right after B
        # AND flip T and last dim for upcoming matmul with Q.
        K_T = K.reshape((B, T, self._num_heads, -1)).permute((0, 2, 3, 1))  # [B, H, d_k, T]

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

        # Add (residual) and norm.
        O_res = in_ + O
        in_ = self._layernorm_1(O_res)

        # Positional feed forward:
        # ------------------------
        O = self._positional_ff(in_)

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
