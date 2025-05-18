import importlib.machinery
import importlib.util
import torch

module_name = "blanchotian_embedding"
file_path = "BlanchotianEmbedding.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
embedding_module = importlib.util.module_from_spec(spec)
loader.exec_module(embedding_module)

BlanchotianEmbedding = embedding_module.BlanchotianEmbedding


def test_embedding_output_shape():
    emb = BlanchotianEmbedding(num_tokens=5, dim=3)
    tokens = torch.randint(0, 5, (2, 4))
    out = emb(tokens)
    assert out.shape == (*tokens.shape, 3)
