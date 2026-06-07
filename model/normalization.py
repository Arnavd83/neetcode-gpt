import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        u = np.mean(x)
        var = np.sum((x - u) ** 2)/ len(x)
        eps = 1e-5
        layer_norm = ((x - u) / (np.sqrt(var + eps))) * gamma + beta
        print(f"u: {u}\nvar: {var}\neps: {eps}")
        return np.round(layer_norm, 5)
