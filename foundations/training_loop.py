import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        samples, features = X.shape[0], X.shape[1]
        w, b = np.zeros(features), 0

        for i in range(epochs):
            # Forward pass
            # (samples, features) * (features,) (samples)
            y_hat = X @ w + b # y_hat -> (samples)
            mse = np.mean((y_hat - y) ** 2)

            # Backwards pass
            dl_dw = (2/samples) * (X.T @ (y_hat - y))
            dl_db = (2/samples) * np.sum(y_hat - y)

            # Updates
            w, b = w - lr * dl_dw, b - lr * dl_db
        

        return (np.round(w, 5), np.round(b, 5))


    '''from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Setup (all done once before training)
        model = MLP()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for X_batch, y_batch in loader:          # inner loop over batches
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()'''
