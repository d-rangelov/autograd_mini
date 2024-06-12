import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

class Tensor:
    """
    A class that represents a tensor in the computation graph.

    Attributes:
        data (np.ndarray): The data of the tensor.
        grad (np.ndarray, optional): The gradient of the tensor. Defaults to None.
        ctx (FunctionCtx, optional): The context for the tensor. Keeps track of how the tensor was generated (e.g. parents, generating function, etc.) Defaults to None.

    Methods:
        backward()
            Performs backward pass to compute gradients.
        dot(y)
            Performs dot product with another tensor "y".
        relu()
            Applies ReLU activation function.
        log_softmax()
            Applies LogSoftmax function.
        nll_loss(y_true)
            Calculates the negative log likelihood loss.
    """

    def __init__(self,
                  data: Optional[Union[np.ndarray, List[float]]]=None,
                  ctx: Optional["FunctionCtx"] = None,
                  shape: Optional[Tuple[int, ...]] = None,
                  init_method: Optional[str] = None):
        
        # Initialize tensor
        if data is not None:
            self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        elif shape is not None and init_method is not None:
            self.data = self.initialize(shape, init_method)
        else:
            raise ValueError("Either 'data' or 'shape' and 'init_method' must be provided.")
        
        self.grad: Optional[np.ndarray] = None
        self.ctx: Optional["FunctionCtx"] = ctx

    def backward(self):
        # Reached last tensor in the graph
        if self.ctx is None:
            return

        # Check if you are at a leaf Tensor
        if self.grad is None:
            # Set gradient to 1's
            self.grad = np.ones_like(self.data)

        # (Special Case: Unary op - e.g. ReLU)
        if len(self.ctx.parents) == 1:
            (p1,) = self.ctx.parents
            # Initialize gradient if it doesn't exist for accumulation
            if p1.grad is None:
                p1.grad = np.zeros_like(p1.data, dtype=np.float32)
            # Accumulate gradient
            p1.grad += self.ctx.backward(self.grad)
            p1.backward()
            return

        # Extract both parents
        p1, p2 = self.ctx.parents
        # Compute parent gradients with kernel's backward fn
        p1_grad, p2_grad = self.ctx.backward(self.grad)

        # Initialize gradients if they don't exist for accumulation
        if p1.grad is None:
            p1.grad = np.zeros_like(p1.data, dtype=np.float32)
        if p2.grad is None:
            p2.grad = np.zeros_like(p2.data, dtype=np.float32)

        # Accumulate gradients
        p1.grad += p1_grad
        p2.grad += p2_grad
        # Continue backward pass
        p1.backward()
        p2.backward()

    # TODO: Implement other weight initializations (e.g. Xavier)
    def initialize(self, shape: Tuple[int, ...], method: str) -> np.ndarray:
        fan_in = np.prod(shape[:-1])
        fan_out = np.prod(shape)
        
        if method == "kaiming":
            bound = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(low=-bound, high=bound, size=shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def dot(self, y: "Tensor") -> "Tensor":
        return Dot(self, y).forward()

    def relu(self) -> "Tensor":       
        return ReLU(self).forward()

    def log_softmax(self) -> "Tensor":
        return LogSoftmax(self).forward()

    def nll_loss(self, y_true: np.ndarray) -> "Tensor":
        return NLLLoss(self, y_true).forward()

    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __str__(self) -> str:
        return f"{self.data}"

    def __repr__(self) -> str:
        return f"{self.data}"

    def __len__(self) -> int:
        return len(self.data)


class FunctionCtx(ABC):
    """
    Abstract base class for all functions in the computation graph.

    Attributes:
        parents (Tuple[Tensor, ...]): Tensors that are inputs to the function.
        saved_tensors (List[np.ndarray]): List of tensors saved for the backward pass.
    
    Methods:
        save_for_backward(*x)
            Saves input tensors for use in the backward pass.
        
        forward()
            Performs the forward computation.
        
        backward(grad_output)
            Performs the backward computation.
    """

    def __init__(self, *tensors: Tensor):
        self.parents: Tuple[Tensor, ...] = tensors
        self.saved_tensors: List[np.ndarray] = []

    def save_for_backward(self, *x: np.ndarray):
        self.saved_tensors.extend(x)

    @abstractmethod
    def forward(self) -> Tensor:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        pass


class Mul(FunctionCtx):
    def forward(self) -> Tensor:
        x, y = self.parents
        # Save the values not the references to the parents
        self.save_for_backward(x.data, y.data)
        result = Tensor(x.data * y.data, ctx=self)
        return result

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        x, y = self.saved_tensors
        return y * grad_output, x * grad_output


class Dot(FunctionCtx):

    def forward(self) -> Tensor:
        input, weight = self.parents
        self.save_for_backward(input.data, weight.data)
        result = Tensor(input.data.dot(weight.data), ctx=self)
        return result

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        input, weight = self.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = input.T.dot(grad_output)
        return grad_input, grad_weight


class ReLU(FunctionCtx):

    def forward(self) -> Tensor:
        (input,) = self.parents
        self.save_for_backward(input.data)
        result = Tensor(np.maximum(input.data, 0), ctx=self)
        return result

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        (x,) = self.saved_tensors
        grad_input = grad_output * (x >= 0)
        return grad_input


class LogSoftmax(FunctionCtx):

    def forward(self) -> Tensor:
        (x,) = self.parents
        # Numerical Stability
        x_max = np.max(x.data, axis=-1, keepdims=True)
        x_normalized = x.data - x_max

        log_softmax = x_normalized - np.log(np.sum(np.exp(x_normalized), axis=-1, keepdims=True))
        self.save_for_backward(log_softmax)
        return Tensor(log_softmax, ctx=self)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        (log_softmax,) = self.saved_tensors
        # Revert log_softmax to softmax
        softmax = np.exp(log_softmax)
        grad_input = grad_output - softmax * np.sum(grad_output, axis=-1, keepdims=True)
        return grad_input


class NLLLoss(FunctionCtx):

    def __init__(self, log_probs: Tensor, y_true: np.ndarray):
        super().__init__(log_probs)
        self.y_true = y_true

    def forward(self) -> Tensor:
        (log_probs,) = self.parents
        y_true = self.y_true
        self.save_for_backward(log_probs.data, y_true)
        # Extract the probabilities for the true labels
        predicted_probs = log_probs.data[np.arange(len(y_true)), y_true]
        return Tensor(-np.sum(predicted_probs, axis=-1) / len(y_true), ctx=self)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        log_probs, y_true = self.saved_tensors
        grad_input = np.zeros_like(log_probs, dtype=log_probs.dtype)
        grad_input[np.arange(len(y_true)), y_true] = -1 / len(y_true)
        return grad_input * grad_output
