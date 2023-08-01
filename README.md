# scalargrad
An educational autograd engine implementation following [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.  
`scalargrad` implements forward and backward passes over dynamically built computional graphs of scalars.  
That's enough to train neural networks with `scalargrad`
# Examples
- Learning XOR function with 2 layer MLP classifier
```python
from scalargrad.nn import Linear, Sequential, CrossEntropyLoss, SGD, Softmax

# define dataset.
x = [
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0],
]
y = [1, 0, 0, 1]

# define model, loss function and optimizer.
model = Sequential([
    Linear(2, 4, activation='relu'),
    Linear(4, 2, activation='identity'),
])
loss_fn = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.075)

# train for 1000 epochs with batch size of 1.
for _ in range(1000):
    for xi, yi in zip(x, y):
        output = model(xi)
        target = yi
        optim.zero_grad()
        loss = loss_fn([output], [target])
        loss.backward()
        optim.step()

# predict softmax probabilities for each sample.
softmax = Softmax()
for xi, yi in zip(x, y):
    logits = model(xi)
    pred = softmax(logits)
    pred = [out.data for out in pred]
    print(f'x: {xi}')
    print(f'y: {yi}')
    print(f'pred: {pred}')
    print()


# result.
>>> x: [1.0, 1.0]
>>> y: 1
>>> pred: [0.33366935620417804, 0.6663306437958219]

>>> x: [0.0, 1.0]
>>> y: 0
>>> pred: [0.33366935620417804, 0.6663306437958219]

>>> x: [1.0, 0.0]
>>> y: 0
>>> pred: [0.9984033656965609, 0.0015966343034391618]

>>> x: [0.0, 0.0]
>>> y: 1
>>> pred: [0.33366935620417804, 0.6663306437958219]
```
- Linear function forward and backward pass
```python
from scalargrad.node import Node

x0 = Node(0.1, name='x0')
x1 = Node(0.2, name='x1')
w0 = Node(0.3, name='w0')
w1 = Node(0.4, name='w1')
b = Node(0.5, name='b')
# forward pass.
f = (x0 * w0) + (x1 * w1) + b
# backward pass.
f.backward()
# display computational graph.
f.show()

print(f'forward: {f.data}')
print(f'df/dw0: {w0.grad}')
print(f'df/dw1: {w1.grad}')

# result.
>>> forward: 0.61
>>> df/dw0: 0.1
>>> df/dw1: 0.2
```
![image](examples/linear.png)
