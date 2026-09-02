import numpy as np
import torch
from torch import nn

from elasticai.explorer.platforms.generator.litert_generator import LitertGenerator


def test_generate_tflite_model(tmp_path):
    input_sample = torch.randn(16, 1, 28, 28)
    model = nn.Sequential(
        nn.Conv2d(1, 6, 2), nn.ReLU(), nn.Flatten(), nn.Linear(4374, 4)
    )
    model_path = tmp_path / "model"
    model.eval()
    torch_output = model(input_sample)
    print(torch_output)
    generator = LitertGenerator()
    edge_model = generator.generate(model, model_path, input_sample)

    edge_output = edge_model(input_sample)

    assert np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5)
    assert model_path.with_suffix(".tflite").exists()
    assert model_path.with_suffix(".cpp").exists()
