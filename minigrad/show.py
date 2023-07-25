import tempfile
import graphviz
from PIL import Image


def show_graph(node):
    dot = graphviz.Digraph()
    dot.format = 'png'
    
    nodes = [node]
    visited = set()
    while nodes:
        node = nodes.pop(0)
        if node.name in visited:
            continue
        visited.add(node.name)
        op = node._op
        op_key = f'_op_{node.name}'
        dot.node(node.name, f'{node.name}\ndata={node.data:.4f}\ngrad={node.grad:.4f}', shape='box')
        if op is not None:
            dot.node(op_key, op)
        for ch in node._children:
            dot.node(ch.name, f'{ch.name}\ndata={ch.data:4f}\ngrad={ch.grad:.4f}', shape='box')
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
