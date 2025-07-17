import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.rand(in_features, out_features)
        self.bias = np.random.rand(out_features) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Formula: f(x) = x * W + b
        """
        if self.bias is not None:
            return np.dot(x, self.weight) + self.bias
        else:
            return np.dot(x, self.weight)


class ReLU:
    """
    A simple ReLU activation function.
    Formula: f(x) = max(0, x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Placeholder for ReLU activation logic
        return np.maximum(0, x)


class GELU:
    """
    A simple GELU activation function.
    Formula: f(x) = x * sigmoid(1.702 * x)
    where sigmoid(x) = 1 / (1 + e ^ (-x))
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )


class Softmax:
    """
    A simple Softmax activation function.
    Formula: f(x) = e ^ (x - max(x)) / sum(e ^ (x - max(x)))
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LayerNorm:
    """
    Formula: f(x) = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = np.ones(normalized_shape)
        self.bias = np.zeros(normalized_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(var + self.eps)
        return self.weight * normalized_x + self.bias


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = np.random.rand(vocab_size, embed_dim)

    def forward(self, indices: np.ndarray) -> np.ndarray:
        # Placeholder for embedding lookup logic
        return self.weight[indices]
