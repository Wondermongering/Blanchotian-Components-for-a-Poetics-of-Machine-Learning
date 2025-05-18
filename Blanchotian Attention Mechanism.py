import torch
from torch import nn, einsum
from einops import rearrange

class BlanchotianAttention(nn.Module):
    def __init__(self, dim, heads=8, trace_decay=0.8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.trace_decay = trace_decay
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # Learnable void parameters
        self.void_query = nn.Parameter(torch.randn(1, heads, 1, dim // heads))
        self.void_key = nn.Parameter(torch.randn(1, heads, 1, dim // heads))
        self.void_value = nn.Parameter(torch.randn(1, heads, 1, dim // heads))
        
        # Registered buffer for attention trace persistence
        self.register_buffer('attention_trace', torch.zeros(heads, 1, 1))
        
        # Temperature adjustment parameter
        self.temperature_factor = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x, *, return_attention: bool = False):
        b, n, _ = x.shape
        
        # Project input to queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Add void elements
        void_q = self.void_query.expand(b, -1, -1, -1)
        void_k = self.void_key.expand(b, -1, -1, -1)
        void_v = self.void_value.expand(b, -1, -1, -1)
        q = torch.cat([q, void_q], dim=2)
        k = torch.cat([k, void_k], dim=2)
        v = torch.cat([v, void_v], dim=2)
        
        # Compute attention scores
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * (self.dim ** -0.5)
        
        # Make void attend strongly to itself
        void_index = n
        dots[:, :, void_index, void_index] += 10.0  # Large value to make it dominate
        
        # Compute temperature based on trace
        temperature = 1.0 + self.attention_trace.abs() * self.temperature_factor
        temperature = temperature.clamp(min=1.0)
        
        # Softmax with temperature
        attn = (dots / temperature).softmax(dim=-1)
        
        # Update trace (detached from computation graph)
        with torch.no_grad():
            new_trace = attn[:, :, :-1, :-1].mean(dim=[0, 2, 3])
            self.attention_trace = (self.trace_decay * self.attention_trace + 
                                   (1 - self.trace_decay) * new_trace)
        
        # Compute output
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Only keep the outputs corresponding to the original tokens
        out = out[:, :, :-1, :]
        
        # Rearrange and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attention:
            return out, attn
        return out
