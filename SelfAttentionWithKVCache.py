import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_k=None, past_v=None):
        B, T, D = x.shape  # Batch, Time, Dim

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Append cached keys/values
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)  # (B, H, T_total, Dh)
            v = torch.cat([past_v, v], dim=2)

        # Save new cache
        new_k, new_v = k, v

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, H, T, T_total)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, Dh)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output), new_k.detach(), new_v.detach()

# Example usage:
B, T, D = 1, 1, 64  # Batch size, sequence length, embedding dim
model = SelfAttentionWithKVCache(d_model=64, n_heads=4)
token_embeddings = [torch.randn(B, 1, D) for _ in range(5)]  # Generate 5 tokens one by one

past_k, past_v = None, None
for i, x in enumerate(token_embeddings):
    output, past_k, past_v = model(x, past_k, past_v)
    print(f"Step {i+1}, Output shape: {output.shape}, Cache length: {past_k.shape[2]}")
