import importlib.machinery
import importlib.util
import torch

module_name = "infinite_conversation"
file_path = "The Infinite Conversation.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
conversation_module = importlib.util.module_from_spec(spec)
loader.exec_module(conversation_module)

BlanchotianTransformer = conversation_module.BlanchotianTransformer


def test_transformer_output_shape():
    model = BlanchotianTransformer(dim=16, depth=2, heads=2, mlp_dim=32)
    x = torch.randn(3, 4, 16)
    out = model(x)
    assert out.shape == x.shape
