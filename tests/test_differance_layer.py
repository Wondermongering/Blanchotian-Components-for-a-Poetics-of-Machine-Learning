import importlib.machinery
import importlib.util
import torch

module_name = "differance_layer"
file_path = "Differance Layer.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
differance_module = importlib.util.module_from_spec(spec)
loader.exec_module(differance_module)

DifferanceLayer = differance_module.DifferanceLayer


def test_output_shape():
    layer = DifferanceLayer(dim=8)
    x = torch.randn(2, 4, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_deferral_effect():
    layer = DifferanceLayer(dim=4)
    x1 = torch.randn(1, 2, 4)
    x2 = torch.randn(1, 2, 4)
    _ = layer(x1)
    out2 = layer(x2)
    assert not torch.allclose(out2, x2)
