import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint




class CWNodeAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, combined_flat, in_w, in_b, hidden_w, hidden_b, ln_w, ln_b, out_w, out_b, chunk_size=128):
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(combined_flat, in_w, in_b, hidden_w, hidden_b, ln_w, ln_b, out_w, out_b)
        
        outputs = []
        with torch.no_grad():
            for i in range(0, combined_flat.shape[0], chunk_size):
                chunk = combined_flat[i : i + chunk_size]
                chunk_perm = chunk.transpose(0, 1) # [N, C]
                
                pre_act = chunk_perm.unsqueeze(-1) * in_w.unsqueeze(1) + in_b.unsqueeze(1)
                h_perm = F.gelu(pre_act)
                
                if hidden_w is not None and hidden_w.shape[0] > 0:
                    for l in range(hidden_w.shape[0]):
                        res = h_perm
                        layer_out = torch.matmul(h_perm, hidden_w[l]) + hidden_b[l].unsqueeze(1)
                        
                        mean = layer_out.mean(dim=-1, keepdim=True)
                        var = layer_out.var(dim=-1, keepdim=True, unbiased=False)
                        layer_normed = (layer_out - mean) / torch.sqrt(var + 1e-5)
                        layer_normed = layer_normed * ln_w[l].unsqueeze(1) + ln_b[l].unsqueeze(1)
                        
                        h_perm = F.gelu(layer_normed) + res
                        
                out_perm = (h_perm * out_w.unsqueeze(1)).sum(dim=-1) + out_b.unsqueeze(1)
                outputs.append(out_perm.transpose(0, 1))
                
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output_flat):
        combined_flat, in_w, in_b, hidden_w, hidden_b, ln_w, ln_b, out_w, out_b = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        
        grad_combined = torch.zeros_like(combined_flat)
        grad_in_w = torch.zeros_like(in_w)
        grad_in_b = torch.zeros_like(in_b)
        
        grad_hidden_w = torch.zeros_like(hidden_w) if hidden_w is not None else None
        grad_hidden_b = torch.zeros_like(hidden_b) if hidden_b is not None else None
        grad_ln_w = torch.zeros_like(ln_w) if ln_w is not None else None
        grad_ln_b = torch.zeros_like(ln_b) if ln_b is not None else None
        
        grad_out_w = torch.zeros_like(out_w)
        grad_out_b = torch.zeros_like(out_b)

        for i in range(0, combined_flat.shape[0], chunk_size):
            chunk = combined_flat[i : i + chunk_size].detach().requires_grad_(True)
            
            with torch.enable_grad():
                chunk_perm = chunk.transpose(0, 1)
                pre_act = chunk_perm.unsqueeze(-1) * in_w.unsqueeze(1) + in_b.unsqueeze(1)
                h_perm = F.gelu(pre_act)
                
                if hidden_w is not None and hidden_w.shape[0] > 0:
                    for l in range(hidden_w.shape[0]):
                        res = h_perm
                        layer_out = torch.matmul(h_perm, hidden_w[l]) + hidden_b[l].unsqueeze(1)
                        
                        mean = layer_out.mean(dim=-1, keepdim=True)
                        var = layer_out.var(dim=-1, keepdim=True, unbiased=False)
                        layer_normed = (layer_out - mean) / torch.sqrt(var + 1e-5)
                        layer_normed = layer_normed * ln_w[l].unsqueeze(1) + ln_b[l].unsqueeze(1)
                        
                        h_perm = F.gelu(layer_normed) + res
                        
                out_perm = (h_perm * out_w.unsqueeze(1)).sum(dim=-1) + out_b.unsqueeze(1)
                out_chunk = out_perm.transpose(0, 1)
            
            grad_out_chunk = grad_output_flat[i : i + chunk_size]
            
            inputs = [chunk, in_w, in_b, out_w, out_b]
            if hidden_w is not None:
                inputs.extend([hidden_w, hidden_b, ln_w, ln_b])
                
            grads = torch.autograd.grad(out_chunk, inputs, grad_outputs=grad_out_chunk, retain_graph=False)
            
            with torch.no_grad():
                grad_combined[i : i + chunk_size] = grads[0]
                grad_in_w += grads[1]
                grad_in_b += grads[2]
                grad_out_w += grads[3]
                grad_out_b += grads[4]
                
                if hidden_w is not None:
                    grad_hidden_w += grads[5]
                    grad_hidden_b += grads[6]
                    grad_ln_w += grads[7]
                    grad_ln_b += grads[8]
                    
            del chunk, grad_out_chunk, grads

        return grad_combined, grad_in_w, grad_in_b, grad_hidden_w, grad_hidden_b, grad_ln_w, grad_ln_b, grad_out_w, grad_out_b, None


class SquareMLP(nn.Module):
    def __init__(self, in_features, out_features, width, depth, dtype=torch.float32):
        super().__init__()
        self.width = width
        self.depth = depth
        
        self.in_proj = nn.Linear(in_features, width, dtype=dtype)
        
        if depth > 1:
            self.hidden_layers = nn.ModuleList([
                nn.Linear(width, width, dtype=dtype) for _ in range(depth - 1)
            ])
            self.ln_layers = nn.ModuleList([
                nn.LayerNorm(width, dtype=dtype) for _ in range(depth - 1)
            ])
        else:
            self.hidden_layers = nn.ModuleList()
            self.ln_layers = nn.ModuleList()
            
        self.out_proj = nn.Linear(width, out_features, dtype=dtype)
        self.reset_parameters()
        
    def reset_parameters(self):
        # We initialize with a small std to help with extreme depth stability
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        for layer in self.hidden_layers:
            # Scale initialization for deep residual networks
            nn.init.normal_(layer.weight, mean=0.0, std=0.02 / math.sqrt(max(1, 2 * self.depth)))
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x):
        h = F.gelu(self.in_proj(x))
        for layer, ln in zip(self.hidden_layers, self.ln_layers):
            # Residual connection with LayerNorm
            res = h
            h = F.gelu(ln(layer(h))) + res
        return self.out_proj(h)

def internal_node_forward(
    combined_flat: torch.Tensor,
    in_w: torch.Tensor,
    in_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    hidden_w: torch.Tensor = None,
    hidden_b: torch.Tensor = None,
    ln_w: torch.Tensor = None,
    ln_b: torch.Tensor = None,
) -> torch.Tensor:
    """
    Device-native pure functional internal-node forward. 
    Bypasses MPS compiler deadlock by using F.layer_norm.
    """
    x_perm = combined_flat.transpose(0, 1)  # [N, C]
    h_perm = F.gelu(x_perm.unsqueeze(-1) * in_w.unsqueeze(1) + in_b.unsqueeze(1))

    if hidden_w is not None and hidden_w.shape[0] > 0:
        for l in range(hidden_w.shape[0]):
            res = h_perm
            layer_out = torch.matmul(h_perm, hidden_w[l]) + hidden_b[l].unsqueeze(1)
            layer_normed = F.layer_norm(layer_out, (layer_out.shape[-1],), eps=1e-5)
            layer_normed = layer_normed * ln_w[l].unsqueeze(1) + ln_b[l].unsqueeze(1)
            h_perm = F.gelu(layer_normed) + res

    out_perm = (h_perm * out_w.unsqueeze(1)).sum(dim=-1) + out_b.unsqueeze(1)
    return out_perm.transpose(0, 1)

class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd, dtype=torch.float32):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, dtype=dtype)
        self.c_proj = nn.Linear(n_embd, n_embd, dtype=dtype)

        # Zero-initialize the output projection so that initially
        # this block acts as an identity mapping
        nn.init.zeros_(self.c_proj.weight)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape to treat the single head as the 'num_heads' dimension: [B, 1, T, C]
        q = q.view(B, T, 1, C).transpose(1, 2)
        k = k.view(B, T, 1, C).transpose(1, 2)
        v = v.view(B, T, 1, C).transpose(1, 2)

        # Efficient memory-optimized causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Restore original shape: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)


class CWNodeLayer(nn.Module):
    """
    Connection-Weighted Node Layer utilizing Square MLPs for both
    external (between-node) mixing and internal (per-node) processing.
    """
    def __init__(self, in_features, out_features, w_ext, d_ext, w_int, d_int, chunk_size=128, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.d_ext = d_ext
        self.w_int = w_int
        self.d_int = d_int
        self.chunk_size = chunk_size

        # Attention block (restored from production)
        self.ln_attn = nn.LayerNorm(in_features, dtype=dtype)
        self.attn = SingleHeadAttention(in_features, dtype=dtype)

        # External feature mixer
        
        # 1. External routing pool (Dense Deep Square MLP)
        self.external = SquareMLP(in_features, out_features, w_ext, d_ext, dtype=dtype)
        
        if self.w_int == 0:
            self.register_parameter('in_w', None)
            self.register_parameter('in_b', None)
            self.register_parameter('hidden_w', None)
            self.register_parameter('hidden_b', None)
            self.register_parameter('ln_w', None)
            self.register_parameter('ln_b', None)
            self.register_parameter('out_w', None)
            self.register_parameter('out_b', None)
            return
            
        # 2. Internal processing pool (Independent Per-Node MLPs) stage (batched per-node parameters)
        self.in_w = nn.Parameter(torch.empty((out_features, w_int), dtype=dtype))
        self.in_b = nn.Parameter(torch.empty((out_features, w_int), dtype=dtype))
        
        if d_int > 1:
            self.hidden_w = nn.Parameter(torch.empty((d_int - 1, out_features, w_int, w_int), dtype=dtype))
            self.hidden_b = nn.Parameter(torch.empty((d_int - 1, out_features, w_int), dtype=dtype))
            self.ln_w = nn.Parameter(torch.ones((d_int - 1, out_features, w_int), dtype=dtype))
            self.ln_b = nn.Parameter(torch.zeros((d_int - 1, out_features, w_int), dtype=dtype))
        else:
            self.register_parameter('hidden_w', None)
            self.register_parameter('hidden_b', None)
            self.register_parameter('ln_w', None)
            self.register_parameter('ln_b', None)
            
        self.out_w = nn.Parameter(torch.empty((out_features, w_int), dtype=dtype))
        self.out_b = nn.Parameter(torch.empty((out_features,), dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self):
        # External resets itself in init
        
        if self.w_int == 0:
            return
            
        # Initialize internal Square MLP stage
        nn.init.normal_(self.in_w, mean=0.0, std=1.0)
        nn.init.zeros_(self.in_b)
        
        if self.hidden_w is not None:
            # Residual scaling for hidden layers
            nn.init.normal_(self.hidden_w, mean=0.0, std=1.0 / math.sqrt(self.w_int * 2 * self.d_int))
            nn.init.zeros_(self.hidden_b)
            
        nn.init.normal_(self.out_w, mean=0.0, std=1.0 / math.sqrt(self.w_int))
        nn.init.zeros_(self.out_b)

    def forward(self, x):
        # 1. Causal self-attention (restored)
        x = x + self.attn(self.ln_attn(x))

        # 2. External feature mixing
        # x shape: (B, T, in_features)
        B, T, _ = x.shape
        
        # 1. External stage runs on the GPU (MPS) at full speed
        combined = self.external(x)
        
        if self.w_int == 0:
            return combined
            
        # 2. Internal stage runs device-native via optimized Metal kernels
        flat = combined.reshape(B * T, self.out_features)
        
        out_flat = internal_node_forward(
            flat,
            self.in_w,
            self.in_b,
            self.out_w,
            self.out_b,
            self.hidden_w,
            self.hidden_b,
            self.ln_w,
            self.ln_b,
        )
        return out_flat.view(B, T, self.out_features)

class CWNodeTransformer(nn.Module):
    """
    Decoder-only character-level Transformer featuring 3 CWNodeLayers,
    standard token/position embeddings, and a linear language modeling head.
    """
    def __init__(self, vocab_size, n_layer, n_embd, w_ext, d_ext, w_int, d_int, block_size, dtype=torch.float32):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        
        self.w_ext = w_ext
        self.d_ext = d_ext
        self.w_int = w_int
        self.d_int = d_int
        
        # Token and positional embeddings
        self.wte = nn.Embedding(vocab_size, n_embd, dtype=dtype)
        self.wpe = nn.Embedding(block_size, n_embd, dtype=dtype)
        
        # Stack of CWNodeLayers
        self.layers = nn.ModuleList([
            CWNodeLayer(n_embd, n_embd, w_ext, d_ext, w_int, d_int, dtype=dtype)
            for _ in range(n_layer)
        ])
        
        # Output language modeling head (dense, bias-free, negligible parameter footprint)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, dtype=dtype)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Let SquareMLP handle its own initialization via reset_parameters
            if not getattr(module, 'is_square_mlp_layer', False):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        
        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(pos) # (1, t, n_embd)
        x = tok_emb + pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.lm_head(x) # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Focus on logits of the last step
            logits = logits[:, -1, :].float() / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
