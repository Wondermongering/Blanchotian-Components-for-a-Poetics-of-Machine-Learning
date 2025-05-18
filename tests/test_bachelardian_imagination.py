import importlib.machinery
import importlib.util
import torch

module_name = "bachelardian_imagination"
file_path = "Bachelardian Imagination Space.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
bachelard_module = importlib.util.module_from_spec(spec)
loader.exec_module(bachelard_module)

BachelardianImaginationSpace = bachelard_module.BachelardianImaginationSpace


def test_generation_shape():
    model = BachelardianImaginationSpace(vocab_size=10, dim=8, hidden_dim=16)
    start = torch.zeros(2, dtype=torch.long)
    gen = model.generate(start, steps=4)
    assert gen.shape == (2, 4)


def test_forward_logits_shape():
    model = BachelardianImaginationSpace(vocab_size=15, dim=8, hidden_dim=16)
    tokens = torch.randint(0, 15, (3, 5))
    logits, hidden = model(tokens)
    assert logits.shape == (3, 5, 15)
    assert hidden.shape == (3, 16)

