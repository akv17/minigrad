class Value:

    def __init__(self, data, name=None, _op=None, _children=None):
        self.data = data
        self.name = name
        self.grad = 0.0

        self._backward = lambda: None
        self._children = _children or ()
        self._op = _op
    
    def __repr__(self):
        return f'Value(data={self.data}, name={self.name}, op={self._op})'
    
    def __add__(self, other):
        out = Value(data=self.data + other.data, _children=(self, other), _op='add')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(data=self.data * other.data, _children=(self, other), _op='mul')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
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


def f():
    x1 = Value(2.0, name='x1')
    x2 = Value(0.0, name='x2')

    w1 = Value(0.2, name='w1')
    w2 = Value(0.3, name='w2')

    u1 = x1 * w1
    u1.name = 'u1'
    
    u2 = x2 * w2
    u2.name = 'u2'

    b = Value(0.5, name='b')

    z = u1 + u2
    z.name = 'z'

    y = z + b
    y.name = 'y'
    return y


def f2():
    a = Value(-2.0, name='a')
    b = Value(3.0, name='b')
    d = a * b
    d.name = 'd'
    e = a + b
    e.name = 'e'
    f = d * e
    f.name = 'f'
    return f


def draw(value):
    import tempfile
    import graphviz
    from PIL import Image

    dot = graphviz.Digraph()
    dot.format = 'png'
    
    nodes = [value]
    visited = set()
    while nodes:
        node = nodes.pop(0)
        if node.name in visited:
            continue
        visited.add(node.name)
        op = node._op
        op_key = f'_op_{node.name}'
        dot.node(node.name, f'{node.name}\ndata={node.data}\ngrad={node.grad}', shape='box')
        if op is not None:
            dot.node(op_key, op)
        for ch in node._children:
            dot.node(ch.name, f'{ch.name}\ndata={ch.data}\ngrad={ch.grad}', shape='box')
            nodes.append(ch)
            if op is not None:
                dot.edge(ch.name, op_key)
        if op is not None:
            dot.edge(op_key, node.name)

    
    with tempfile.TemporaryDirectory() as tmp:
        dot.render(filename='g', directory=tmp)
        fp = f'{tmp}/g.png'
        im = Image.open(fp)
    im.show()


def main():
    v = f()
    v.backward()
    draw(v)


if __name__ == '__main__':
    main()
