import importlib.machinery
import importlib.util
import torch

module_name = "blanchot_activation_module"
file_path = "Blanchot-Guided Activation Function.py"
loader = importlib.machinery.SourceFileLoader(module_name, file_path)
spec = importlib.util.spec_from_loader(module_name, loader)
activation_module = importlib.util.module_from_spec(spec)
loader.exec_module(activation_module)

blanchot_activation = activation_module.blanchot_activation


def test_activation_output_shape_and_no_nans():
    x = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    out = blanchot_activation(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
