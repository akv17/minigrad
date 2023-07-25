from minigrad.node import Node


def f():
    x1 = Node(2.0, name='x1')
    x2 = Node(0.0, name='x2')

    w1 = Node(0.2, name='w1')
    w2 = Node(0.3, name='w2')

    u1 = x1 * w1
    u1.name = 'u1'
    
    u2 = x2 * w2
    u2.name = 'u2'

    b = Node(0.5, name='b')

    z = u1 + u2
    z.name = 'z'

    y = z + b
    y.name = 'y'
    return y


def f2():
    a = Node(-2.0, name='a')
    b = Node(3.0, name='b')
    d = a * b
    d.name = 'd'
    e = a + b
    e.name = 'e'
    f = d * e
    f.name = 'f'
    return f


def main():
    n = f()
    n.backward()
    n.show()


if __name__ == '__main__':
    main()
