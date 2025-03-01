class BlanchotianAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # Recursive trace of previous attention patterns
        self.attention_trace = nn.Parameter(torch.zeros(heads, 1, 1))
        
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Standard attention calculation
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.dim ** -0.5)
        
        # "The Double Reading" - attention scores are influenced by their own absence
        absent_mask = 1.0 - torch.eye(n, device=dots.device)[None, None, :, :]
        recursive_dots = dots * absent_mask + dots * self.attention_trace
        
        # "The Essential Solitude" - each token attends partially to void
        void_attention = torch.zeros_like(dots[:, :, :, 0:1])
        dots_with_void = torch.cat([recursive_dots, void_attention], dim=-1)
        
        attn = dots_with_void.softmax(dim=-1)
        
        # Update attention trace for next forward pass
        self.attention_trace = 0.9 * self.attention_trace + 0.1 * attn[:, :, :, :-1].mean(dim=[0, 1])
        
        # Attend to values (including the "void")
        v_with_void = torch.cat([v, torch.zeros_like(v[:, :, 0:1, :])], dim=2)
        out = torch.matmul(attn, v_with_void)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
