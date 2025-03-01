import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume BlanchotianAttention and FeedForward are defined elsewhere
# (as in previous responses)

class BlanchotianTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, discontinuity_factor=0.5, decay_rate=0.01):
        super().__init__()
        self.depth = depth
        self.discontinuity_factor = discontinuity_factor
        self.decay_rate = decay_rate

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BlanchotianAttention(dim, heads=heads),  # Assuming this exists
                FeedForward(dim, mlp_dim)  # Assuming this exists
            ]))

        # Layer-specific weights: (depth, depth + 1)
        # Each row corresponds to a layer, and the columns are weights for
        # all previous layers (including input)
        self.layer_weights = nn.Parameter(torch.ones(depth, depth + 1))

    def forward(self, x):
        layer_outputs = [x]  # Initialize with the input

        for i, (attn, ff) in enumerate(self.layers):
            # "The Infinite Conversation" - each layer speaks to all previous layers
            # Use layer-specific weights
            weights = F.softmax(self.layer_weights[i, :len(layer_outputs)], dim=0)
            weighted_sum = sum(w * output for w, output in zip(weights, layer_outputs))

            # "The Writing of Disaster" - information passes through discontinuity
            # Apply discontinuity factor to the attention output
            attn_out = attn(weighted_sum)
            x = self.discontinuity_factor * attn_out + weighted_sum  # Residual connection
            x = ff(x) + x  # Residual connection
            layer_outputs.append(x)

        # Return weighted sum of all layer outputs
        # Use the weights from the *last* layer for the final combination
        final_weights = F.softmax(self.layer_weights[-1, :], dim=0)
        return sum(w * output for w, output in zip(final_weights, layer_outputs))

    def regularization_loss(self):
        """
        Calculates a regularization loss that encourages layer weights to decay.
        This favors later layers while still allowing earlier layers to contribute.
        """
        # L2 regularization with a decay factor that increases with layer depth
        decay_factors = torch.arange(1, self.depth + 2, device=self.layer_weights.device).float()
        decay_factors = torch.pow(self.decay_rate, self.depth + 1 - decay_factors) #Exponential decay
        return torch.sum(decay_factors * (self.layer_weights ** 2))

# Example FeedForward (replace with your actual implementation)
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
#Dummy Blanchotian Attention
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

# Example Usage and Training Loop (with regularization)
# Assuming you have your data loaders and OrphicEmbeddings ready

# Example Data
batch_size = 4
sequence_length = 10
vocab_size = 1000 # Example
embedding_dim = 128
num_epochs = 2

# Instantiate the model
dim = embedding_dim  # Dimensionality of the embeddings
depth = 6            # Number of layers
heads = 8            # Number of attention heads
mlp_dim = 256        # Feed-forward network hidden dimension
model = BlanchotianTransformer(dim, depth, heads, mlp_dim)

# Dummy input data (replace with your actual data loading)
x = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Optimizer (AdamW is often better than Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x)

    # Example loss function (replace with your actual loss)
    # Here, we're just summing the output for demonstration purposes
    loss = output.sum()

    # Add regularization loss
    reg_loss = model.regularization_loss()
    total_loss = loss + reg_loss

    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Regularization Loss: {reg_loss.item()}")

print("Training complete.")
