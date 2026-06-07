import torch
import torch.nn as nn
from typing import List
import numpy as np


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        curr = x
        dead_list = []
        for layer in model.children():
            curr = layer(curr)
            if isinstance(layer, nn.ReLU):
                num_dead = (curr == 0).all(dim=0).sum().item()
                total = curr.shape[1]
                dead_list.append((float)(num_dead) / (float)(total))
        return dead_list

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        
        if any(val > .5 for val in dead_fractions):
            return 'use_leaky_relu'
        if any(val > .3 for val in dead_fractions):
            return 'reinitialize'
        if np.all(np.diff(dead_fractions) > 0) and np.any(val > .1 for val in dead_fractions):
            return 'reduce_learning_rate'
        if max(dead_fractions) < .1:
            return 'healthy'
        else:
            return 'healthy'
