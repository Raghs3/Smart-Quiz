import numpy as np

LAYER_SIZES = [5, 16, 8, 4, 1]


class DeepNN:
    def __init__(self, layer_sizes=None):
        sizes = layer_sizes or LAYER_SIZES
        self.layer_sizes = list(sizes)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            scale = 1.0 / self.layer_sizes[i] ** 0.5
            W = np.random.uniform(-scale, scale, (self.layer_sizes[i + 1], self.layer_sizes[i]))
            b = np.zeros((self.layer_sizes[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)
        self._acts = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sig_d(self, a):
        return a * (1.0 - a)

    def forward(self, x):
        a = np.asarray(x, dtype=float).reshape(-1, 1)
        self._acts = [a]
        for W, b in zip(self.weights, self.biases):
            a = self._sigmoid(W @ a + b)
            self._acts.append(a)
        return self._acts[-1].ravel()

    def backward(self, y, lr=0.05):
        if not self._acts:
            raise RuntimeError("backward() called before forward(). Call forward() first.")
        target = np.array([[float(np.clip(y, 0.0, 1.0))]])
        delta = self._acts[-1] - target
        for i in reversed(range(len(self.weights))):
            dW = delta @ self._acts[i].T
            db = delta.copy()
            if i > 0:
                delta = (self.weights[i].T @ delta) * self._sig_d(self._acts[i])
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def save(self, path):
        arrays = {f'W{i}': W for i, W in enumerate(self.weights)}
        arrays.update({f'b{i}': b for i, b in enumerate(self.biases)})
        arrays['layer_sizes'] = np.array(self.layer_sizes)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        layer_sizes = data['layer_sizes'].tolist()
        nn = cls(layer_sizes)
        nn.weights = [data[f'W{i}'] for i in range(len(layer_sizes) - 1)]
        nn.biases = [data[f'b{i}'] for i in range(len(layer_sizes) - 1)]
        return nn
