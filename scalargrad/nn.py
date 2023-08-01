import random

from .util import uid
from .scalar import Scalar


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
        w = w or [random.uniform(-0.1, 0.1) for _ in range(self.size)]
        self._w = [Scalar(wi, name=f'w{i}@{self.name}') for i, wi in enumerate(w)]
        b = b or random.uniform(-0.1, 0.1)
        self._b = Scalar(b, name=f'b@{self.name}')

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
                for i in range(self.size_out)
            ]
    
    def parameters(self):
        return [p for n in self._neurons for p in n.parameters()]


class Sequential:

    def __init__(self, modules, name=None, do_initialize=False):
        self.modules = modules
        self.name = name or f'sequential@{uid()}'
        if do_initialize:
            self.initialize()
    
    def __repr__(self):
        return f'Sequential(name={self.name}, size={len(self.modules)}))'

    def __call__(self, x):
        for mod in self.modules:
            x = mod(x)
        return x

    def initialize(self, modules_data=None):
        if modules_data is not None:
            assert len(modules_data) == len(self.modules)
            for mod_data, mod in zip(modules_data, self.modules):
                mod.initialize(**mod_data)
        else:
            for mod in self.modules:
                mod.initialize()

    def parameters(self):
        return [p for m in self.modules for p in m.parameters()]


class Softmax:

    def __init__(self, name=None):
        self.name = name or f'softmax@{uid()}'

    def __repr__(self):
        return f'Softmax(name={self.name})'

    def __call__(self, x):
        assert isinstance(x, list)
        assert x
        call_id = uid()
        exps = [xi.exp().set_name(f'exp{i}@{call_id}@{self.name}') for i, xi in enumerate(x)]
        norm = sum(exps).set_name(f'norm@{call_id}@{self.name}')
        x = [(ei / norm).set_name(f'prob{i}@{call_id}@{self.name}') for i, ei in enumerate(exps)]
        return x

    def initialize():
        pass

    def parameters():
        pass


class CrossEntropyLoss:

    def __init__(self, name=None):
        self.name = name or f'cross-entropy@{uid()}'
        self.softmax = Softmax()

    def __repr__(self):
        return f'CrossEntropyLoss(name={self.name})'

    def __call__(self, outputs, targets):
        assert outputs
        assert len(outputs) == len(targets)
        assert len(set(len(o) for o in outputs)) == 1  # same number of classes for each output.
        call_id = uid()
        logs = []
        for i, (out, target) in enumerate(zip(outputs, targets)):
            out = self.softmax(out)
            pred = out[target]
            log = pred.log().set_name(f'log{i}@{call_id}@{self.name}')
            logs.append(log)
        loss = (-1.0 * (sum(logs) / len(logs))).set_name(f'loss@{call_id}@{self.name}')
        return loss

    def initialize():
        pass

    def parameters():
        pass


class MSELoss:

    def __init__(self, name=None):
        self.name = name or f'mse@{uid()}'

    def __repr__(self):
        return f'MSELoss(name={self.name})'

    def __call__(self, outputs, targets):
        assert outputs
        assert len(outputs) == len(targets)
        assert set(len(o) for o in outputs) == {1}  # same number of classes for each output.
        outputs = [out[0] for out in outputs]
        call_id = uid()
        losses = [
            ((out - target) ** 2).set_name(f'loss{i}@{call_id}@{self.name}')
            for i, (out, target) in enumerate(zip(outputs, targets))
        ]
        loss = (sum(losses) / len(losses)).set_name(f'loss@{call_id}@{self.name}')
        return loss

    def initialize():
        pass

    def parameters():
        pass


class SGD:
    
    def __init__(self, parameters, lr, momentum=None, name=None):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.name = name or f'sgd@{uid()}'
        self._step = 0
        self._momentum_buffer = [0.0] * len(self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        self._step += 1
        for pi, p in enumerate(self.parameters):
            g = p.grad
            if self.momentum is not None:
                if self._step > 1:
                    b = self.momentum * self._momentum_buffer[pi] + g
                else:
                    b = g
                self._momentum_buffer[pi] = b
                g = b
            p.data -= self.lr * g
