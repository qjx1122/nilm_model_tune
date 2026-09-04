import torch
from src.model import NILMTransformer

def test_forward():
    m = NILMTransformer(input_dim=1, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    x = torch.randn(8, 64, 1)
    y = m(x)
    assert y.shape == (8,)
