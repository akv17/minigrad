import os
from abc import ABC, abstractmethod

_COMPUTE = None


class Scalar:

    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f'Scalar({self.value})'
    
    def __add__(self, b):
        b = self._wrap_maybe(b)
        res = _COMPUTE.add(self.value, b.value)
        return res
    
    def __sub__(self, b):
        b = self._wrap_maybe(b)
        res = _COMPUTE.sub(self.value, b.value)
        return res

    def __mul__(self, b):
        b = self._wrap_maybe(b)
        res = _COMPUTE.mul(self.value, b.value)
        return res
    
    def __truediv__(self, b):
        b = self._wrap_maybe(b)
        res = _COMPUTE.div(self.value, b.value)
        return res
    
    def __pow__(self, b):
        b = self._wrap_maybe(b)
        res = _COMPUTE.pow(self.value, b.value)
        return res
    
    def __neg__(self):
        res = _COMPUTE.mul(self.value, -1.0)
        return res

    __radd__ = __add__
    
    __rmul__ = __mul__
    
    def __rtruediv__(self, a):
        a = self._wrap_maybe(a)
        res = _COMPUTE.div(a.value, self.value)
        return res
    
    def __rsub__(self, a):
        a = self._wrap_maybe(a)
        res = _COMPUTE.sub(a.value, self.value)
        return res

    def exp(self):
        res = _COMPUTE.exp(self.value)
        return res

    def log(self):
        res = _COMPUTE.log(self.value)
        return res
    
    def _wrap_maybe(self, value):
        ob = type(self)(value) if not isinstance(value, type(self)) else value
        return ob


class ICompute(ABC):
    
    @abstractmethod
    def add(self, a, b): pass

    @abstractmethod
    def sub(self, a, b): pass
    
    @abstractmethod
    def mul(self, a, b): pass
    
    @abstractmethod
    def div(self, a, b): pass
    
    @abstractmethod
    def pow(self, a, b): pass
    
    @abstractmethod
    def exp(self, a): pass
    
    @abstractmethod
    def log(self, a): pass


class PyCompute(ICompute):

    def __init__(self):
        import math
        self._math = math

    def add(self, a, b):
        return a + b
    
    def sub(self, a, b):
        return a - b
    
    def mul(self, a, b):
        return a * b
    
    def div(self, a, b):
        return a / b

    def pow(self, a, b):
        return a ** b
    
    def exp(self, a):
        return self._math.exp(a)
    
    def log(self, a):
        return self._math.log(a)


class TorchCompute(ICompute):
    
    def __init__(self):
        import torch
        self._torch = torch
    
    def add(self, a, b):
        return (self._as_tensor(a) + self._as_tensor(b)).item()
    
    def sub(self, a, b):
        return (self._as_tensor(a) - self._as_tensor(b)).item()
    
    def mul(self, a, b):
        return (self._as_tensor(a) * self._as_tensor(b)).item()
    
    def div(self, a, b):
        return (self._as_tensor(a) / self._as_tensor(b)).item()

    def pow(self, a, b):
        return (self._as_tensor(a) ** self._as_tensor(b)).item()
    
    def exp(self, a):
        return (self._as_tensor(a).exp()).item()
    
    def log(self, a):
        return (self._as_tensor(a).log()).item()

    def _as_tensor(self, v):
        return self._torch.tensor(v, dtype=self._torch.float64, requires_grad=False)
    
    __radd__ = add
    
    __rmul__ = mul
    
    __rtruediv__ = div


class NumpyCompute(ICompute):
    
    def __init__(self):
        import numpy
        self._np = numpy
    
    def add(self, a, b):
        return (self._as_array(a) + self._as_array(b)).item()
    
    def sub(self, a, b):
        return (self._as_array(a) - self._as_array(b)).item()
    
    def mul(self, a, b):
        return (self._as_array(a) * self._as_array(b)).item()
    
    def div(self, a, b):
        return (self._as_array(a) / self._as_array(b)).item()

    def pow(self, a, b):
        return (self._as_array(a) ** self._as_array(b)).item()
    
    def exp(self, a):
        return self._np.exp(a).item()
    
    def log(self, a):
        return self._np.log(a).item()

    def _as_array(self, v):
        return self._np.array(v, dtype=self._np.float64)
    
    __radd__ = add
    
    __rmul__ = mul
    
    __rtruediv__ = div


def _init_compute():
    global _COMPUTE
    if _COMPUTE is None:
        type_ = os.getenv('COMPUTE', 'py')
        dispatch = {
            'py': PyCompute,
            'torch': TorchCompute,
            'numpy': NumpyCompute,
        }
        try:
            ob = dispatch[type_]()
            _COMPUTE = ob
        except KeyError:
            msg = f'unknown compute: {type_} ({list(dispatch)})'
            raise Exception(msg)


_init_compute()
