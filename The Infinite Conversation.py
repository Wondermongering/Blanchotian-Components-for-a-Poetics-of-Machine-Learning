import torch
import torch.nn as nn
import torch.nn.functional as F

class BlanchotianTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, max_memory=5,
                 discontinuity_factor=0.6, noise_factor=0.02,
                 init_temp=1.0, tau=0.5, use_gumbel=True, decay_rate=0.01):
        super().__init__()
        self.depth = depth
        self.max_memory = max_memory  # Limited memory
        self.discontinuity_factor = discontinuity_factor  # Control discontinuity
        self.noise_factor = noise_factor  # Control writing noise
        self.use_gumbel = use_gumbel # Toggle for Gumbel-Softmax
        self.decay_rate = decay_rate

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BlanchotianAttention(dim, heads=heads),
                FeedForward(dim, mlp_dim),
                nn.Parameter(torch.ones(1))  # Per-layer weight (for Gumbel)
            ]))

        # Layer-specific weights, but initialized with a temperature schedule
        self.layer_weights = nn.Parameter(torch.linspace(init_temp, 0, depth + 1))
        self.tau = tau

        # For decaying regularization
        self.register_buffer('decay_factors', torch.pow(self.decay_rate, torch.arange(depth + 1, 0, -1).float()))

    def forward(self, x):
        memory_buffer = [x]

        for i, (attn, ff, layer_weight) in enumerate(self.layers):
            # Fragmentary Memory Management
            if len(memory_buffer) > self.max_memory:
                memory_buffer = memory_buffer[-self.max_memory:]

            # Weighted conversation with history
            if self.use_gumbel:
                weights = F.gumbel_softmax(self.layer_weights[:len(memory_buffer)] * layer_weight, tau=self.tau, hard=False)
            else:
                weights = F.softmax(self.layer_weights[:len(memory_buffer)], dim=0)

            conversation = sum(w * mem for w, mem in zip(weights, memory_buffer))

            # Discontinuous transformation
            attn_out = attn(conversation)
            x = self.discontinuity_factor * attn_out + conversation
            x = ff(x) + x

            # Noisy memory update
            memory_buffer.append(x + torch.randn_like(x) * self.noise_factor)

        # Final fragmentary composition (using last layer's individual weight if gumbel)
        if self.use_gumbel:
           final_weights = F.gumbel_softmax(self.layer_weights * self.layers[-1][2], tau=self.tau, hard = False)
        else:
           final_weights =  F.softmax(self.layer_weights, dim=0)
        return sum(w * mem for w, mem in zip(final_weights, memory_buffer))


    def regularization_loss(self):
      """
        Calculates a regularization loss that encourages layer weights
        to decay (favoring later layers).
      """
      return torch.sum(self.decay_factors * (self.layer_weights ** 2))

#Dummy Blanchotian Attention and FeedForward
class BlanchotianAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: torch.reshape(t, (b, n, self.heads, -1)).permute(0, 2, 1, 3), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.dim ** -0.5)
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = torch.reshape(out.permute(0, 2, 1, 3), (b, n, -1))

        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Example Usage and Training Loop (with regularization)
    # Assuming you have your data loaders and OrphicEmbeddings ready

    # Example Data
    batch_size = 4
    sequence_length = 10
    vocab_size = 1000  # Example
    embedding_dim = 128
    num_epochs = 2

    # Instantiate the model
    dim = embedding_dim
    depth = 6
    heads = 8
    mlp_dim = 256
    model = BlanchotianTransformer(dim, depth, heads, mlp_dim)

    # Dummy input data
    x = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x)

        # Example loss function (replace)
        loss = output.sum()

        # Add regularization loss
        reg_loss = model.regularization_loss()
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1}, Loss: {loss.item()}, Regularization Loss: {reg_loss.item()}"
        )

    print("Training complete.")
