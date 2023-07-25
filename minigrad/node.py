class Node:

    def __init__(self, data, name=None, _op=None, _children=None):
        self.data = data
        self.name = name
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
