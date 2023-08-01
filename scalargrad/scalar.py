import sys
import math

from .util import uid

sys.setrecursionlimit(10000)


class Scalar:

    def __init__(self, data, name=None, _op=None, _children=None):
        self.data = data
        self.name = name or f'scalar@{uid()}'
        self.grad = 0.0

        self._backward = lambda: None
        self._children = _children or ()
        self._op = _op
    
    def __repr__(self):
        return f'Scalar(data={self.data:.4f}, name={self.name}, op={self._op})'
    
    def __add__(self, other):
        other = self._as_scalar(other)
        out = Scalar(data=self.data + other.data, _children=(self, other), _op='add')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = self._as_scalar(other)
        out = Scalar(data=self.data - other.data, _children=(self, other), _op='sub')
        
        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = self._as_scalar(other)
        res = other - self
        return res
    
    def __mul__(self, other):
        other = self._as_scalar(other)
        out = Scalar(data=self.data * other.data, _children=(self, other), _op='mul')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._as_scalar(other)
        out = Scalar(data=self.data / other.data, _children=(self, other), _op='div')
        
        def _backward():
            self.grad += 1.0 / other.data * out.grad
            other.grad += -self.data / (other.data ** 2) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = self._as_scalar(other)
        return other / self

    def __pow__(self, value):
        out = Scalar(data=self.data ** value, _children=(self,), _op='pow')
        
        def _backward():
            n = value
            self.grad += (n * self.data ** (n-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Scalar(data=-self.data, _children=(self,), _op='neg')
        
        def _backward():
            self.grad += -out.grad
        
        out._backward = _backward
        return out
    
    __radd__ = __add__
    
    __rmul__ = __mul__
    
    def exp(self):
        data = math.exp(self.data)
        out = Scalar(data=data, _children=(self,), _op='exp')

        def _backward():
            self.grad += data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Scalar(data=math.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += 1 / self.data * out.grad
        
        out._backward = _backward
        return out

    def sqrt(self):
        out = self ** 0.5
        return out
    
    def relu(self):
        data = self.data if self.data > 0.0 else 0.0
        out = Scalar(data=data, _children=(self,), _op='relu')

        def _backward():
            value = 1.0 if self.data > 0.0 else 0.0
            self.grad += value * out.grad
        
        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1 / (1 + math.exp(-self.data))
        out = Scalar(data=data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += data * (1 - data) * out.grad
        
        out._backward = _backward
        return out

    def identity(self):
        out = Scalar(data=self.data, _children=(self,), _op='identity')

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        visited = set()
        nodes_sorted = []

        def _traverse(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for ch in node._children:
                _traverse(ch)
            nodes_sorted.append(node)
        
        _traverse(self)
        for node in reversed(nodes_sorted):
            node._backward()

    def set_name(self, name):
        self.name = name
        return self

    def render(self):
        from .util import render_graph
        render_graph(self)

    def _as_scalar(self, other):
        other = Scalar(other) if not isinstance(other, type(self)) else other
        return other
