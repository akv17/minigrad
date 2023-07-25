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
        linear.name = f'linear@{call_id}@{self.name}'
        act = getattr(linear, self.activation)()
        act.name = f'act@{call_id}@{self.name}'
        return act

    def initialize(self, w=None, b=None):
        if w is not None:
            assert len(w) == self.size
        w = w or [random.uniform(-1.0, 1.0) for _ in range(self.size)]
        self._w = [Node(wi, name=f'w{i}@{self.name}') for i, wi in enumerate(w)]
        b = b or random.uniform(-1.0, 1.0)
        self._b = Node(b, name=f'b@{self.name}')

    def parameters(self):
        return [*self._w, self._b]


class Linear:

    @classmethod
    def from_torch(cls, module, activation='identity', name=None):
        size_out = module.weight.data.shape[0]
        size_in = module.weight.data.shape[1]
        ob = cls(size_in=size_in, size_out=size_out, activation=activation, name=name)
        w = module.weight.data
        b = module.bias.data
        neurons = []
        for i, (wi, bi) in enumerate(zip(w, b)):
            neuron = Neuron(size=size_in, activation=activation, name=f'neuron{i}@{ob.name}')
            neuron.initialize(w=wi.tolist(), b=bi.item())
            neurons.append(neuron)
        ob.initialize(neurons)
        return ob

    def __init__(self, size_in, size_out, activation='identity', name=None):
        self.size_in = size_in
        self.size_out = size_out
        self.activation = activation
        self.name = name or f'linear@{uid()}'

        self._neurons = None
        self.initialize()
    
    def __repr__(self):
        return f'Linear(name={self.name}, size_in={self.size_in}, size_out={self.size_out}))'

    def __call__(self, x):
        assert len(x) == self.size_in
        out = [n(x) for n in self._neurons]
        return out

    def initialize(self, neurons=None):
        self._neurons = neurons
        if self._neurons is None:
            self._neurons = [
                Neuron(
                    size=self.size_in,
                    activation=self.activation,
                    name=f'neuron{i}@{self.name}'
                )
                for i in range(self.size_in)
            ]
    
    def parameters(self):
        return [p for n in self._neurons for p in n.parameters()]
