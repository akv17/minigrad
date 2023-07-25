import math
from uuid import uuid4


class Node:

    def __init__(self, data, name=None, _op=None, _children=None):
        self.data = data
        self.name = name or f'@{str(uuid4())[:8]}'
        self.grad = 0.0

        self._backward = lambda: None
        self._children = _children or ()
        self._op = _op
    
    def __repr__(self):
        return f'Node(data={self.data}, name={self.name}, op={self._op})'
    
    def __add__(self, other):
        out = Node(data=self.data + other.data, _children=(self, other), _op='add')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Node(data=self.data * other.data, _children=(self, other), _op='mul')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = self * (other ** -1)
        return out

    def __pow__(self, value):
        out = Node(data=self.data ** value, _children=(self,), _op='pow')
        
        def _backward():
            n = value
            self.grad += (n * self.data ** (n-1)) *  out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        data = math.exp(self.data)
        out = Node(data=data, _children=(self,), _op='exp')

        def _backward():
            self.grad += data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Node(data=math.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += 1 / self.data * out.grad
        
        out._backward = _backward
        return out

    def sqrt(self):
        out = self ** 0.5
        return out
    
    def relu(self):
        data = self.data if self.data > 0.0 else 0.0
        out = Node(data=data, _children=(self,), _op='relu')

        def _backward():
            value = 1.0 if self.data > 0.0 else 0.0
            self.grad += value * out.grad
        
        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1 / (1 + math.exp(-self.data))
        out = Node(data=data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += data * (1 - data) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        visited = set()
        nodes_sorted = []

        def _traverse(node):
            if node.name in visited:
                return
            visited.add(node.name)
            for ch in node._children:
                _traverse(ch)
            nodes_sorted.append(node)
        
        _traverse(self)
        for node in reversed(nodes_sorted):
            node._backward()

    def show(self):
        from .show import show_graph
        show_graph(self)
