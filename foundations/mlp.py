import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:

    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array (n)
        # weights: list of 2D weight matrices ()
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        curr_l = x
        for i in range(len(weights)):
            curr_w, curr_b = weights[i], biases[i]
            curr_l = np.maximum(0, (curr_l @ np.squeeze(curr_w) + curr_b))
        return curr_l

