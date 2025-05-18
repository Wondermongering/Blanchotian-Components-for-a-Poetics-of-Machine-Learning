import importlib.machinery
import importlib.util
import torch
from torch import nn

module_name = "blanchotian_layer_norm"
file_path = "Blanchotian Layer Normalization.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
ln_module = importlib.util.module_from_spec(spec)
loader.exec_module(ln_module)

BlanchotianLayerNorm = ln_module.BlanchotianLayerNorm


def test_layer_norm_output_shape():
    ln = BlanchotianLayerNorm(4)
    x = torch.randn(2, 4)
    out = ln(x)
    assert out.shape == x.shape


def test_layer_norm_outlier_preservation():
    ln = BlanchotianLayerNorm(4, solitude_factor=1.0)
    std_ln = nn.LayerNorm(4)
    std_ln.weight.data = ln.weight.data.clone()
    std_ln.bias.data = ln.bias.data.clone()
    x = torch.tensor([[0.0, 0.0, 0.0, 1000.0]])
    out_std = std_ln(x)
    out_bl = ln(x)
    assert out_bl[0, 3].abs() > out_std[0, 3].abs()


def test_layer_norm_matches_standard_without_outliers():
    ln = BlanchotianLayerNorm(4)
    std_ln = nn.LayerNorm(4)
    std_ln.weight.data = ln.weight.data.clone()
    std_ln.bias.data = ln.bias.data.clone()
    x = torch.randn(3, 4)
    out_std = std_ln(x)
    out_bl = ln(x)
    assert torch.allclose(out_bl, out_std, atol=1e-5)
