# Autograd Mini 

This is a mini deep learning library built from scratch using Python and Numpy. The main motivation behind the project is to use it as a learning tool for developing deeper understanding of how modern libs like Pytorch work under the hood.

It's a hobby project of mine inspired by **Andrej Karpathy's** [micrograd](https://github.com/karpathy/micrograd) and **George Hotz'** (a.k.a. **geohot**) [tinygrad](https://github.com/tinygrad/tinygrad). In terms of complexity, `Autograd Mini` falls somewhere between the above-mentioned projects. Compared to `micrograd`, it makes working with tensors and creating Feed Forward Neural Networks (FNNs) easier, but it is nowhere near as capable *(or as complex)* as `tinygrad`. 

If you are completely new to the topic of Neural Networks and how they work, I **HIGHLY** recommend watching [this outstanding course](https://www.youtube.com/watch?v=VMj-3S1tku0) by Andrej, where he implements a neural network from scratch and provides an exceptional explaination of backpropagation. After watching Andrej's course, you might revisit this current repository to find out how, what you've learned, can be incorporated in a deep learning library and a tiny autograd engine.

## Current Features

Currently the library supports the following concepts:

- **Tensors**: Multi-dimensional arrays with automatic differentiation capabilities.
- **Operations**: Basic operations - multiply, dot product, ReLU, Softmax, etc.
- **Compute Graph and Automatic Differentiation**: Supports the construction of a compute graph and perfoms automatic differentiation during the backward pass of the network.
- **SGD Optimizer**:  Simple gradient descent optimizer to update model parameters. The optimizer supports similar APIs to Pytorch with methods such as `step()` and `zero_grad()`, making it easy to integrate with the rest of your deep learning pipeline.
- **Basic Dataset & DataLoader**: Provides a convenient way to load and iterate over your dataset. The Dataset class holds the features and labels, while the DataLoader class handles batch processing and optional data shuffling between epochs.

## Installation

To use this library, simply clone the repository and install the necessary dependencies.

```bash
git clone  
cd  
pip install -r requirements.txt
```

## Usage Example
- You can find a jupyter notebook with a complete classification example on the MNIST dataset [here](https://github.com/d-rangelov/autograd_mini/).

- For a quick overview of the main functionalities of the library, check the code below:

```python
from autograd_mini.tensor import Tensor
from autograd_mini.data import Dataset, DataLoader
from autograd_mini.optim import Optim as SGD

class MNISTNet:
    
    def __init__(self):
        self.W1 = Tensor(shape=(784, 256), init_method="kaiming")
        self.W2 = Tensor(shape=(256, 10), init_method="kaiming")

    def params(self):
        return [self.W1, self.W2]

    def __call__(self, x: Tensor) -> Tensor:
        x = x.dot(self.W1)
        x = x.relu()
        x = x.dot(self.W2)
        x = x.log_softmax()
        return x


EPOCHS = 10

# Load the data
(x_train, y_train), (x_test, y_test) = ... # Loading data

# Create data loader for easy shuffle and batch splitting
train_dataset = Dataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Init the Model and optimizer
model = MNISTNet()
optim = SGD(lr = 0.001, params = model.params())

# Training Loop
epoch_losses = []
for epoch in range(EPOCHS):
    losses = []
    # Training Loss
    for Xb, yb in train_dataloader:
        out = model(Xb)
        loss = out.nll_loss(yb)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(loss.data)

```
## Future TODOs
Currently, the library supports only a few basic operations and network architectures. There are still a lot of "low-hanging fruits" that are still missing (e.g. other activation functions, `.ones()`, `.zeros()`, `.diag()`, etc.), so I might gradually start adding these.

However, what I am more excited about is implementing additional ops from scratch (e.g. RNN, CNN, GRU, etc.), as well as providing GPU support, so that the library allows for more complex architectures and faster training. 