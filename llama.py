import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

class ModelArgs:
    bos_token_id = 1
    eos_token_id = 2
    d_model: int = 48
    hidden_dim: int = 1024
    seq_len: int = 8192 * 2
    n_heads: int = 12
    n_layers: int = 72
    n_kv_heads: int = 3
    d_ff: int = 48
    max_batch_len: int = 8
    vocab_size: int = 2000
    dropout: float = 0.1

def precompute_theta_pos_frequencies(d_model, seq_len, theta: float = 10000.0):
    theta_num = torch.arange(0, d_model, 2, dtype=dtype, device=device)
    theta = 1.0 / (theta ** (theta_num / d_model)).to(dtype=dtype)

    if seq_len > 10000:
        chunk_size = 5000
        freqs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            m = torch.arange(start, end, dtype=dtype, device=device)
            freqs.append(torch.outer(m, theta))
        freqs = torch.cat(freqs, dim=0)
    else:
        m = torch.arange(seq_len, dtype=dtype, device=device)
        freqs = torch.outer(m, theta)
    freqs_complex = torch.polar(torch.ones_like(freqs, dtype=torch.float32), freqs.to(dtype=torch.float32))
    return freqs_complex.to(dtype=dtype, device=device)


def apply_rotary_embeddings(x, freqs_complex):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)).to(device=device)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated).flatten(3).to(device=device,  dtype=dtype)
    return x_out.type_as(x).to(device=device,  dtype=dtype)

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    x = x[:, :, :, None, :] 
    x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) 
    return x

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model)).to(device=device,  dtype=dtype)

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).to(device=device,  dtype=dtype)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class FeedForward_SiLU(nn.Module):
    def __init__(self, d_ff, hidden_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_ff, hidden_dim, bias=False).to(device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, d_ff, bias=False).to(device=device, dtype=dtype)
        self.w3 = nn.Linear(d_ff, hidden_dim, bias=False).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout).to(device=device, dtype=dtype)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return self.dropout(x)
    
class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model).to(device=device,  dtype=dtype)
        self.dropout = nn.Dropout(dropout).to(device=device,  dtype=dtype)
        self.d_model = d_model

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = x * math.sqrt(self.d_model)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout).to(device=device,  dtype=dtype)
        self.rms_norm = RMSNorm(d_model)

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(x))
        x = self.rms_norm(x)
        return x

class AdvanceAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_batch_len, max_seq_len, dropout):
        super().__init__()
        self.batch_size = max_batch_len
        self.seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_d_model = d_model // n_heads
        self.n_heads_q = n_heads
        self.n_reps = self.n_heads_q // n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, n_heads * self.head_d_model, bias=False).to(device=device, dtype=dtype)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_d_model, bias=False).to(device=device, dtype=dtype)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_d_model, bias=False).to(device=device, dtype=dtype)
        self.wo = nn.Linear(n_heads * self.head_d_model, d_model, bias=False).to(device=device, dtype=dtype)

        self.register_buffer('cache_k', torch.zeros((max_batch_len, max_seq_len, n_kv_heads, self.head_d_model)).to(device=device, dtype=dtype))
        self.register_buffer('cache_v', torch.zeros((max_batch_len, max_seq_len, n_kv_heads, self.head_d_model)).to(device=device, dtype=dtype))

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(self, x, k, v, start_pos, freq_complex):
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(k)
        xv = self.wv(v)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_d_model)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_d_model)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_d_model)

        xq = apply_rotary_embeddings(xq, freq_complex)
        xk = apply_rotary_embeddings(xk, freq_complex)

        with torch.no_grad():
            self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        keys = repeat_kv(keys, self.n_reps)
        values = repeat_kv(values, self.n_reps)

        xq = xq.transpose(1, 2).to(device=device, dtype=dtype)
        keys = keys.transpose(1, 2).to(device=device, dtype=dtype)
        values = values.transpose(1, 2).to(device=device, dtype=dtype)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / torch.tensor(math.sqrt(self.head_d_model), device=device, dtype=dtype)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq).to(device=device, dtype=dtype)

        output = torch.matmul(scores, values).to(device=device, dtype=dtype)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, hidden_dim, n_kv_heads, max_batch_len, max_seq_len, dropout):
        super().__init__()
        self.attention = AdvanceAttention(d_model, n_heads, n_kv_heads, max_batch_len, max_seq_len, dropout)
        self.feed_forward = FeedForward_SiLU(d_ff, hidden_dim, dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)]).to(device=device, dtype=dtype)

    def forward(self, x, start_pos, freq_complex):
        
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, start_pos, freq_complex))
        x = self.residual_connection[1](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers.to(device=device, dtype=dtype)
        self.norm = RMSNorm(d_model).to(device=device, dtype=dtype)

    def clear_cache(self):
        for layer in self.layers:
            layer.attention.clear_cache()  # Only clear the cache for the `attention` attribute

    def forward(self, x, start_pos, freq_complex, src_mask, tgt_mask):
        if start_pos == 0:
            self.clear_cache()
        for layer in self.layers:
            x = layer(x, start_pos, freq_complex)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size).to(device=device,  dtype=dtype)

    def forward(self, x):
        return self.linear(x)

class LLama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_embedding = InputEmbedding(config.d_model, config.vocab_size, config.dropout).to(device=device,  dtype=dtype)
        self.decoder = Decoder(
            config.d_model,
            nn.ModuleList([DecoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.hidden_dim,
                config.n_kv_heads,
                config.max_batch_len,
                config.seq_len,
                config.dropout).to(device=device,  dtype=dtype)
                for _ in range(config.n_layers)])).to(device=device,  dtype=dtype)
        self.projection = ProjectionLayer(config.d_model, config.vocab_size).to(device=device,  dtype=dtype)

        self.freq_complex = precompute_theta_pos_frequencies(config.d_model // config.n_heads, config.seq_len * 2)

    def clear_all_cache(self):
        self.decoder.clear_cache()

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, attention_mask=None) -> torch.Tensor:
        src = input_ids.to(device=device,  dtype=torch.long)
        seq_len = src.size(1)
        freqs_complex = self.freq_complex[start_pos:start_pos + seq_len]
        
        if attention_mask is None:
            if seq_len > 1:
                attention_mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)
        
        src_embedded = self.input_embedding(src)
        src_encoded = self.decoder(src_embedded, start_pos, freq_complex=freqs_complex, src_mask=attention_mask, tgt_mask=attention_mask)
        output = self.projection(src_encoded)
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        
        src = input_ids.to(device=device, dtype=torch.long)
        batch_size = src.size(0)
        generated = src

        with torch.no_grad():
            for _ in range(max_len):
                output = self.forward(generated, start_pos)
                logits = output

                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")

                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                start_pos += 1

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        return generated

config = ModelArgs()
model = LLama(config)

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])


generated_ids = model.generate(input_ids, start_pos=0, max_len=20, temperature=0.7, top_k=50, eos_token_id=2)
print(generated_ids)