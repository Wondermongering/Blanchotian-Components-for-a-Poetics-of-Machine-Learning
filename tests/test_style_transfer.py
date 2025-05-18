import importlib.machinery
import importlib.util
import torch

module_name = 'blanchot_style_transfer'
file_path = 'Blanchotian_Style_Transfer.py'
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)

blanchotian_style_transfer = module.blanchotian_style_transfer


def test_style_transfer_runs():
    token_ids = [1, 2, 3]
    style_vec = torch.randn(32)
    out = blanchotian_style_transfer(token_ids, vocab_size=50, style_vector=style_vec)
    assert out.shape[0] == len(token_ids)
