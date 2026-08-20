import torch 

from torchforge.models import get_model_spec, model_names 

def test_all_models_execute():
    for name in model_names():
        spec = get_model_spec(name)
        model = spec.factory().eval()
        inputs = spec.inputs()
        with torch.inference_mode():
            result = model(*inputs)
        assert isinstance(result, torch.Tensor)
        assert result.numel() > 0

def test_model_registry_is_stable():
    assert {"elementwise", "mlp", "matmul", "conv"}.issubset(model_names())

