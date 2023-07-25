from .util import uid


class Sequential:

    def __init__(self, modules, name=None, do_initialize=True):
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
