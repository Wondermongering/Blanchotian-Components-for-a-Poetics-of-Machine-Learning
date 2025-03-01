import torch
import torch.nn as nn
import torch.nn.functional as F

class ParadoxicalConsensus(nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_models))

    def forward(self, outputs):
        # outputs: List of [batch_size, dim] tensors
        return torch.einsum('m,mbd->bd', F.softmax(self.weights, dim=0), torch.stack(outputs))

class UnavowableCommunity(nn.Module):
    def __init__(self, models, dim, disagreement_weight=0.3, use_kl_divergence=False,
                 history_decay=0.9, history_weight=0.2, use_gumbel_consensus=False, tau=0.5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.dim = dim
        self.disagreement_weight = disagreement_weight # Or nn.Parameter
        self.use_kl_divergence = use_kl_divergence
        self.history_decay = history_decay
        self.history_weight = history_weight
        self.use_gumbel_consensus = use_gumbel_consensus
        self.tau = tau # for gumbel softmax
        self.num_models = len(models)

        # Hypercube-based disagreement calculation
        self.disagreement_proj = nn.Linear(dim, dim)

        # Consensus network
        self.consensus_net = ParadoxicalConsensus(len(models))

        # Paradoxical Gating (refined)
        self.paradox_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),  # Input is just the consensus now
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )

        # Historical Disagreement (use register_buffer for persistence)
        self.register_buffer('historical_disagreement', torch.zeros(1, dim))

    def forward(self, x):
        # model_outputs: List of [batch_size, dim] tensors
        model_outputs = [model(x) for model in self.models]

        # --- Consensus Calculation ---
        if self.use_gumbel_consensus:
            gumbel_weights = F.gumbel_softmax(self.consensus_net.weights, tau = self.tau, hard = False)
            consensus = torch.einsum('m,mbd->bd', gumbel_weights, torch.stack(model_outputs))
        else:
            consensus = self.consensus_net(model_outputs)


        # --- Disagreement Calculation (Hypercube) ---
        hypercube = torch.stack(model_outputs)  # [num_models, batch_size, dim]
        #No need to compute centroid: consensus already computed.
        disagreements = hypercube - consensus.unsqueeze(0)  # [num_models, batch_size, dim]

        if self.use_kl_divergence:
            #Ensure outputs are probabilities
            hypercube = F.softmax(hypercube, dim = -1)
            disagreement_energy = torch.zeros_like(consensus)
            for i in range(self.num_models):
                for j in range(i + 1, self.num_models):
                    kl_div = F.kl_div(hypercube[i].log(), hypercube[j], reduction='none').sum(-1)
                    kl_div += F.kl_div(hypercube[j].log(), hypercube[i], reduction='none').sum(-1)
                    disagreement_energy += kl_div.unsqueeze(-1) #Keep the dimension
        else: #Use energy
            disagreement_energy = torch.mean(disagreements**2, dim=0)  # [batch_size, dim]

        fresh_disagreement = self.disagreement_proj(disagreement_energy)

        # --- Historical Disagreement ---
        historical_component = self.history_weight * self.historical_disagreement
        total_disagreement = fresh_disagreement + historical_component
        # Update historical disagreement (detached from computation graph)
        self.historical_disagreement = self.history_decay * self.historical_disagreement + (1 - self.history_decay) * fresh_disagreement.detach()

        # --- Paradoxical Gating ---
        gate = self.paradox_gate(consensus)  # Gate based on consensus *only*

        # --- Final Output ---
        return consensus + gate * self.disagreement_weight * torch.tanh(total_disagreement)


    def diversity_regularization(self, model_outputs):
        """Calculates a regularization loss based on disagreement (negative cosine similarity)."""
        if self.use_kl_divergence: #If using KL, we have already computed it
            return -torch.mean(torch.stack(model_outputs))

        hypercube = torch.stack(model_outputs)
        centroid = torch.mean(hypercube, dim=0)
        disagreements = hypercube - centroid.unsqueeze(0)
        disagreement_energy = torch.mean(disagreements**2, dim=0)
        return -torch.mean(disagreement_energy) #Minimize energy, maximize disagreement


# Example Usage (assuming you have defined 'models', 'dim', and a suitable loss function)
# Create some dummy models
class DummyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x):
      return self.linear(x)

dim = 64
models = [DummyModel(dim) for _ in range(3)]
community = UnavowableCommunity(models, dim)

# Dummy input
x = torch.randn(32, dim)  # Batch size 32

# Forward pass
output = community(x)

# Example training loop (with diversity regularization)
optimizer = torch.optim.AdamW(community.parameters(), lr=1e-4)
num_epochs = 10
reg_strength = 0.01

for epoch in range(num_epochs):
    optimizer.zero_grad()
    model_outputs = [model(x) for model in community.models] #Need to compute here for regularization
    output = community(x)

    # Replace this with your actual task loss
    task_loss = torch.mean(output**2)  # Dummy task loss

    # Diversity regularization
    diversity_loss = community.diversity_regularization(model_outputs)

    total_loss = task_loss + reg_strength * diversity_loss
    total_loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Task Loss: {task_loss.item()}, Diversity Loss {diversity_loss.item()}")
