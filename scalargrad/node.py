import sys
import math

from .scalar import Scalar as _S
from .util import uid

sys.setrecursionlimit(10000)


class Node:

    def __init__(self, data, name=None, _op=None, _children=None):
        self.data = data
        self.name = name or f'node@{uid()}'
        self.grad = 0.0

        self._backward = lambda: None
        self._children = _children or ()
        self._op = _op
    
    def __repr__(self):
        return f'Node(data={self.data:.4f}, name={self.name}, op={self._op})'
    
    def __add__(self, other):
        other = self._maybe_wrap_with_node(other)
        out = Node(data=_S(self.data) + _S(other.data), _children=(self, other), _op='add')
        
        def _backward():
            self.grad = _S(self.grad) + 1.0 * _S(out.grad)
            other.grad = _S(other.grad) + 1.0 * _S(out.grad)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self._maybe_wrap_with_node(other)
        out = Node(data=_S(self.data) * _S(other.data), _children=(self, other), _op='mul')
        
        def _backward():
            self.grad = _S(self.grad) + _S(other.data) * _S(out.grad)
            other.grad = _S(other.grad) + _S(self.data) * _S(out.grad)
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._maybe_wrap_with_node(other)
        out = Node(data=_S(self.data) / _S(other.data), _children=(self, other), _op='div')
        
        def _backward():
            # f = a / b = a * 1 / b
            # da/df = 1 / b
            # db/df = -a / b**2 
            # g = 1 / b
            # db/dg = b * 0 - 1 * 1 / b**2
            # db/dg = -1 / b**2
            self.grad = _S(self.grad) + (1.0 / _S(other.data) * _S(out.grad))
            other.grad = _S(other.grad) + ((-_S(self.data) / (_S(other.data) ** 2)) * _S(out.grad))

        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        print('----------------------->')
        other = self._maybe_wrap_with_node(other)
        return other / self

    def __pow__(self, value):
        out = Node(data=_S(self.data) ** value, _children=(self,), _op='pow')
        
        def _backward():
            n = value
            self.grad = _S(self.grad) + (_S(n) * _S(self.data) ** (_S(n)-1)) * _S(out.grad)
        
        out._backward = _backward
        return out
    
    __radd__ = __add__
    
    __rmul__ = __mul__
    
    def exp(self):
        data = _S(self.data).exp()
        out = Node(data=data, _children=(self,), _op='exp')

        def _backward():
            self.grad = _S(self.grad) + _S(data) * _S(out.grad)
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Node(data=_S(self.data).log(), _children=(self,), _op='log')

        def _backward():
            self.grad = _S(self.grad) + (1 / _S(self.data) * _S(out.grad))
        
        out._backward = _backward
        return out

    def sqrt(self):
        out = _S(self) ** 0.5
        return out
    
    def relu(self):
        data = self.data if self.data > 0.0 else 0.0
        out = Node(data=data, _children=(self,), _op='relu')

        def _backward():
            value = 1.0 if self.data > 0.0 else 0.0
            self.grad = _S(self.grad) + _S(value) * _S(out.grad)
        
        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1 / (1 + _S(-self.data).exp())
        out = Node(data=data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad = _S(self.grad) + (_S(data) * (1 - _S(data)) * _S(out.grad))
        
        out._backward = _backward
        return out

    def identity(self):
        out = Node(data=self.data, _children=(self,), _op='identity')

        def _backward():
            self.grad = _S(self.grad) + _S(out.grad)

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

    def set_name(self, name):
        self.name = name
        return self

    def show(self):
        from .util import show_graph
        show_graph(self)

    def _maybe_wrap_with_node(self, other):
        other = Node(other) if not isinstance(other, type(self)) else other
        return other
