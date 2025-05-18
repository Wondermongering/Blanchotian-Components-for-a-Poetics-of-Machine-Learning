import importlib.machinery
import importlib.util
import torch

loader = importlib.machinery.SourceFileLoader(
    "visualizer", "Void Attention Visualizer.py")
spec = importlib.util.spec_from_loader("visualizer", loader)
vis_module = importlib.util.module_from_spec(spec)
loader.exec_module(vis_module)

loader_attn = importlib.machinery.SourceFileLoader(
    "blanchotian_attention", "Blanchotian Attention Mechanism.py")
spec_attn = importlib.util.spec_from_loader("blanchotian_attention", loader_attn)
attn_module = importlib.util.module_from_spec(spec_attn)
loader_attn.exec_module(attn_module)

BlanchotianAttention = attn_module.BlanchotianAttention


def test_plot_void_attention_runs():
    attn = BlanchotianAttention(dim=8, heads=2)
    x = torch.randn(1, 3, 8)
    fig = vis_module.plot_void_attention(attn, x, head=0)
    assert hasattr(fig, "axes")
