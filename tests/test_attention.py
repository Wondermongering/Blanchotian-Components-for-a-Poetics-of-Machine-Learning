import importlib.machinery
import importlib.util
import torch

# Dynamically load the module with a space in the filename
module_name = "blanchotian_attention"
file_path = "Blanchotian Attention Mechanism.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
attention_module = importlib.util.module_from_spec(spec)
loader.exec_module(attention_module)

BlanchotianAttention = attention_module.BlanchotianAttention


def test_attention_forward_pass_output_shape():
    attn = BlanchotianAttention(dim=16, heads=4)
    x = torch.randn(2, 5, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_trace_updates():
    attn = BlanchotianAttention(dim=8, heads=2)
    x = torch.randn(1, 3, 8)
    initial_trace = attn.attention_trace.clone()
    _ = attn(x)
    updated_trace = attn.attention_trace
    assert not torch.allclose(initial_trace, updated_trace)
