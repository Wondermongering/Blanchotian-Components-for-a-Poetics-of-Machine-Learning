import importlib.machinery
import importlib.util
import torch

module_name = "unavowable_moe"
file_path = "Unavowable Mixture-of-Experts.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
moe_module = importlib.util.module_from_spec(spec)
loader.exec_module(moe_module)

UnavowableMixtureOfExperts = moe_module.UnavowableMixtureOfExperts

class DummyExpert(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)
    def forward(self, x):
        return self.lin(x)


def test_moe_forward_shape():
    experts = [DummyExpert(16) for _ in range(4)]
    moe = UnavowableMixtureOfExperts(experts, dim=16)
    x = torch.randn(2, 16)
    out = moe(x)
    assert out.shape == torch.Size([2, 16])


def test_disagreement_loss_scalar():
    experts = [DummyExpert(8) for _ in range(3)]
    moe = UnavowableMixtureOfExperts(experts, dim=8)
    x = torch.randn(1, 8)
    _ = moe(x)
    loss = moe.disagreement_loss()
    assert loss.dim() == 0
