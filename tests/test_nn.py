import numpy as np
import os
import tempfile
from nn import DeepNN, LAYER_SIZES


def test_forward_output_shape():
    nn = DeepNN()
    out = nn.forward(np.zeros(5))
    assert out.shape == (1,)


def test_forward_output_range():
    nn = DeepNN()
    for _ in range(20):
        x = np.random.uniform(0, 1, size=5)
        out = nn.forward(x)
        assert 0.0 <= out[0] <= 1.0


def test_backward_changes_weights():
    np.random.seed(0)
    nn = DeepNN()
    W0_before = nn.weights[0].copy()
    x = np.array([1.0, 0.5, 0.6, 0.2, 0.4])
    nn.forward(x)
    nn.backward(0.8)
    assert not np.allclose(nn.weights[0], W0_before)


def test_backward_reduces_loss():
    np.random.seed(42)
    nn = DeepNN()
    x = np.array([1.0, 0.3, 0.5, 0.4, 0.3])
    target = 0.7
    losses = []
    for _ in range(100):
        out = nn.forward(x)
        losses.append((out[0] - target) ** 2)
        nn.backward(target, lr=0.1)
    assert losses[-1] < losses[0]


def test_save_load_roundtrip():
    np.random.seed(7)
    nn = DeepNN()
    x = np.array([1.0, 0.5, 0.6, 0.2, 0.4])
    out_before = nn.forward(x)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        nn.save(path)
        nn2 = DeepNN.load(path)
        out_after = nn2.forward(x)
        assert np.allclose(out_before, out_after)
        assert nn2.layer_sizes == LAYER_SIZES
    finally:
        os.unlink(path)


def test_custom_layer_sizes():
    nn = DeepNN([3, 8, 1])
    out = nn.forward(np.ones(3))
    assert out.shape == (1,)
