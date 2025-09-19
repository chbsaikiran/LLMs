# decoder_only_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# -------------------------
# Helper functions
# -------------------------
def top_k_logits(logits: torch.Tensor, k: int):
    """Keep only top-k logits (set others to -inf). logits: (..., vocab)"""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

def top_p_logits(logits: torch.Tensor, p: float):
    """Nucleus (top-p) filtering.
    logits: (..., vocab). Returns logits with low-prob tokens masked to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens with cumulative prob above p
    sorted_mask = cumulative_probs > p
    # Shift mask to keep at least one token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Set masked logits to -inf
    sorted_logits = torch.where(sorted_mask, torch.full_like(sorted_logits, -1e10), sorted_logits)
    # Unsort
    logits_filtered = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return logits_filtered

# -------------------------
# Modules
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                            # (B, T, C)
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None    # (T, T) boolean mask or None
    ):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # If past exists, concatenate keys/values on time dim for caching
        if layer_past is not None:
            # past_key, past_value shapes: (B, H, T_past, d)
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        # Save present for caching
        present = (k, v)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, T_q, T_k)
        # Causal mask: ensure that position i cannot attend to j > i (future)
        # If attn_mask provided, it should already include causalness or other masks.
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        else:
            # Build causal mask automatically for current sequence length
            Tq = q.size(2)
            Tk = k.size(2)
            device = attn_scores.device
            causal_mask = torch.tril(torch.ones((Tq, Tk), device=device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)   # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, present

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(n_heads, d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, layer_past=None, attn_mask=None):
        # Self-attention (with residual)
        a, present = self.attn(self.ln1(x), layer_past=layer_past, attn_mask=attn_mask)
        x = x + a
        # Feed-forward (with residual)
        x = x + self.ff(self.ln2(x))
        return x, present

# -------------------------
# Decoder-only Transformer
# -------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)

        # Output head (tie weights with token embedding if desired)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.LongTensor,                                  # (B, T)
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None                 # not used extensively here
    ):
        """
        Returns:
          logits: (B, T, vocab_size)
          present_key_values: list of (k, v) for each layer -> shapes (B, H, T_total, d_head)
        """
        B, T = input_ids.size()
        device = input_ids.device

        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        present_key_values = []
        for i, block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[i]
            x, present = block(x, layer_past=layer_past, attn_mask=None)
            present_key_values.append(present)

        x = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab)
        return logits, present_key_values

# -------------------------
# Generation / sampling helper
# -------------------------
@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int = 0,
    top_p: float = 0.0,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None
):
    """
    Auto-regressive generation using KV cache to avoid recomputing past attention.
    input_ids: (B, T_init)
    Returns: (B, T_init + max_new_tokens)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = input_ids.size(0)
    input_ids = input_ids.to(device)

    # We'll maintain past_key_values: list of length n_layers, each is (k, v) with shapes (B, H, T_past, d)
    past_key_values = None

    generated = input_ids
    cur_len = generated.size(1)

    for step in range(max_new_tokens):
        # For efficiency with cache, feed only the last token at each step after first pass.
        if past_key_values is None:
            # first forward pass with the entire context
            logits, past_key_values = model(generated, past_key_values=None)
            # take logits for last position
            next_token_logits = logits[:, -1, :]   # (B, vocab)
        else:
            # pass only the last token and provide past keys
            last_tok = generated[:, -1:].to(device)
            logits, past_key_values = model(last_tok, past_key_values=past_key_values)
            next_token_logits = logits[:, -1, :]

        # apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / (temperature + 1e-8)

        # top-k / top-p filtering
        if top_k > 0:
            next_token_logits = top_k_logits(next_token_logits, k=top_k)
        if top_p > 0.0 and top_p < 1.0:
            next_token_logits = top_p_logits(next_token_logits, p=top_p)

        # sample or greedy
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # append
        generated = torch.cat([generated, next_tokens], dim=1)

        # early stop if all batches generated eos_token_id
        if eos_token_id is not None:
            if (next_tokens.squeeze(-1) == eos_token_id).all():
                break

        # safety: cap sequence length at model.max_seq_len
        if generated.size(1) > model.max_seq_len:
            # you may want to shift/remove earlier tokens in practice; here we just stop
            break

    return generated

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Minimal runtime example (toy sizes)
    vocab_size = 50257  # same as GPT-2 tokenizer
    model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=512, n_layers=6, n_heads=8, max_seq_len=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # You can use a HF tokenizer for converting strings -> ids:
    # pip install transformers
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        prompt = "Once upon a time"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # (1, T)
    except Exception:
        # fallback: numeric example (random ids)
        print("transformers not available: using random tokens as prompt")
        input_ids = torch.randint(0, vocab_size, (1, 8), dtype=torch.long)

    # Move to device
    input_ids = input_ids.to(device)

    # Generate
    out_ids = generate(
        model,
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id if 'tokenizer' in locals() else None,
        device=device
    )
    if 'tokenizer' in locals():
        print("Generated text:")
        print(tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True))
    else:
        print("Generated token ids:", out_ids)
