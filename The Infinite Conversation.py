class BlanchotianTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depths = depth
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BlanchotianAttention(dim, heads=heads),
                FeedForward(dim, mlp_dim)
            ]))
            
        # "The Work of Fire" - each layer's output is partially preserved
        self.layer_weights = nn.Parameter(torch.ones(depth + 1))
            
    def forward(self, x):
        layer_outputs = [x]
        
        for attn, ff in self.layers:
            # "The Infinite Conversation" - each layer speaks to all previous layers
            weights = F.softmax(self.layer_weights[:len(layer_outputs)], dim=0)
            weighted_sum = sum(w * output for w, output in zip(weights, layer_outputs))
            
            # "The Writing of Disaster" - information passes through discontinuity
            x = attn(weighted_sum) + weighted_sum
            x = ff(x) + x
            layer_outputs.append(x)
            
        # Return weighted sum of all layer outputs
        final_weights = F.softmax(self.layer_weights, dim=0)
        return sum(w * output for w, output in zip(final_weights, layer_outputs))
