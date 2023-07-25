import random

from .util import uid
from .node import Node


class Neuron:
    
    def __init__(self, size, activation='identity', name=None):
        self.size = size
        self.activation = activation
        self.name = name or f'neuron@{uid()}'
        
        self._w = None
        self._b = None
        self.initialize()
    
    def __repr__(self):
        return f'Neuron(name={self.name}, size={self.size}))'
        
    def __call__(self, x):
        assert len(x) == self.size
        call_id = uid()
        linear = sum([xi * wi for xi, wi in zip(x, self._w)]) + self._b
        linear.name = f'linear@{call_id}#{self.name}'
        act = getattr(linear, self.activation)()
        act.name = f'act@{call_id}#{self.name}'
        return act

    def initialize(self, w=None, b=None):
        if w is not None:
            assert len(w) == self.size
        w = w or [random.uniform(-1.0, 1.0) for _ in range(self.size)]
        self._w = [Node(wi, name=f'w{i}#{self.name}') for i, wi in enumerate(w)]
        b = b or random.uniform(-1.0, 1.0)
        self._b = Node(b, name=f'b#{self.name}')

    def parameters(self):
        return [*self._w, self._b]


class Linear:
    pass
