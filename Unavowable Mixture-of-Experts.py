import torch
import torch.nn as nn
import torch.nn.functional as F

class DisagreementGate(nn.Module):
    """Simple gating network producing soft selection over experts."""
    def __init__(self, dim, num_experts, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # x: (batch, dim)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class UnavowableMixtureOfExperts(nn.Module):
    """Mixture-of-Experts emphasising disagreement among experts."""
    def __init__(self, experts, dim, disagreement_weight=0.1, hidden_dim=None):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.dim = dim
        self.gate = DisagreementGate(dim, self.num_experts, hidden_dim)
        self.disagreement_weight = disagreement_weight
        self.last_expert_outputs = None

    def forward(self, x):
        # x: (batch, dim)
        gate_probs = self.gate(x)  # (batch, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch, num_experts, dim)
        self.last_expert_outputs = expert_outputs
        output = torch.einsum('bn,bnd->bd', gate_probs, expert_outputs)
        return output

    def disagreement_loss(self):
        if self.last_expert_outputs is None:
            raise RuntimeError("Call forward before disagreement_loss")
        outputs = F.normalize(self.last_expert_outputs, dim=-1)
        pairwise = torch.einsum('bnd,bmd->bnm', outputs, outputs)
        n = outputs.size(1)
        mask = torch.triu(torch.ones(n, n, device=outputs.device), diagonal=1)
        loss = (pairwise * mask).sum() / mask.sum()
        return self.disagreement_weight * loss


# Example usage
if __name__ == "__main__":
    class DummyExpert(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
        def forward(self, x):
            return torch.tanh(self.fc(x))

    dim = 32
    experts = [DummyExpert(dim) for _ in range(4)]
    moe = UnavowableMixtureOfExperts(experts, dim)
    x = torch.randn(8, dim)
    out = moe(x)
    loss = moe.disagreement_loss()
    print(out.shape, loss.item())
