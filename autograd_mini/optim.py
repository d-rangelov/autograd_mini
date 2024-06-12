import numpy as np
from typing import List
from .tensor import Tensor

class Optim:
    """
    Optimizer class to handle gradient descent updates.

    Attributes:
        lr (float): Learning rate for the optimizer.
        params (List[Tensor]): List of tensors that will be updated.

    Methods:
        step():
            Updates the parameters using gradient descent.
        zero_grad():
            Resets the gradients of all parameters to zero.
    """

    def __init__(self, lr: float = 0.001, params: List['Tensor']=[]):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad, dtype=np.float32)
