import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        curr = x
        with torch.no_grad(): 
            for layer in model.children():
                output = layer(curr)
                dic = {
                    'mean': round(output.mean().item(), 4),
                    'std': round(output.std().item(), 4),
                    'dead_fraction': round((output <= 0).all(dim=0).float().mean().item(), 4)
                }
                curr = output
                if isinstance(layer, nn.Linear):
                    stats.append(dic)
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        # Zero out the gradients
        # Run a forward pass of everything to get predictions
        # Calculate loss using mse module
        # use backwards to get to get gradients using pytorch graph
        # Iterate through graph and get the gradients for every single linear layer
        # calculate and store stats

        model.zero_grad()
        y_hat = model(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)
        loss.backward()

        stats = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                curr_grad = layer.weight.grad
                dic = {
                    'mean': round(curr_grad.mean().item(), 4),
                    'std': round(curr_grad.std().item(), 4),
                    'norm': round(curr_grad.norm().item(), 4)
                }
                stats.append(dic)
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        # 1. Dead neurons
        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        # 2. Exploding gradients (any layer)
        for layer in gradient_stats:
            if layer['norm'] > 1000:
                return 'exploding_gradients'
        
        # 3. Vanishing gradients (last layer only)
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        # 4. Activation std check (all layers)
        for layer in activation_stats:
            if layer['std'] < 0.1:
                return 'vanishing_gradients'
            if layer['std'] > 10.0:
                return 'exploding_gradients'
        
        # 5. All good
        return 'healthy'

