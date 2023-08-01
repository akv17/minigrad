import unittest
from parameterized import parameterized

from scalargrad.node import Node
from tests.util import require_torch

torch = require_torch()


class TestOps(unittest.TestCase):
    FP_PRECISION = 5

    @parameterized.expand([
        ('pow:0', '__pow__', 3.0, 2),
        ('pow:1', '__pow__', 3.0, -1),
        ('pow:2', '__pow__', 3.0, 0),
        ('pow:3', '__pow__', 3.0, 0.5),
        ('pow:4', '__pow__', 3.0, 4.3),
        ('pow:5', '__pow__', 1.0, 10),
        
        ('neg:0', '__neg__', 2.31),
        ('neg:1', '__neg__', -2.31),
        ('neg:2', '__neg__', 0.0),
        
        ('exp:0', 'exp', 2),
        ('exp:1', 'exp', 0),
        ('exp:2', 'exp', 1),
        ('exp:3', 'exp', 3.33),
        ('exp:4', 'exp', -4.11),
        
        ('log:0', 'log', 2.7),
        ('log:1', 'log', 1e-5),
        ('log:2', 'log', 1.0),
        ('log:3', 'log', 42),
        
        ('sqrt:0', 'sqrt', 9.0),
        ('sqrt:1', 'sqrt', 2.72422),
        ('sqrt:2', 'sqrt', 1e-5),
        ('sqrt:3', 'sqrt', 121.221),
        
        ('relu:0', 'relu', -1.0),
        ('relu:1', 'relu', 0.0),
        ('relu:2', 'relu', 1e-5),
        
        ('sigmoid:0', 'sigmoid', 0.0),
        ('sigmoid:1', 'sigmoid', -0.99),
        ('sigmoid:2', 'sigmoid', 0.99),
        ('sigmoid:3', 'sigmoid', 13.45),
        ('sigmoid:4', 'sigmoid', -54.0121),
    ])
    def test_unary_op(self, name, op, x, *args):
        t_x = torch.tensor(x, requires_grad=True, dtype=torch.float64)
        t_c = getattr(t_x, op)(*args)
        t_c.backward()

        m_x = Node(x, 'x')
        m_c = getattr(m_x, op)(*args)
        m_c.backward()
        self.assertAlmostEqual(1.0, m_c.grad, places=self.FP_PRECISION)
        self.assertAlmostEqual(t_c.item(), m_c.data, places=self.FP_PRECISION, msg='data')
        self.assertAlmostEqual(t_x.grad.item(), m_x.grad, places=self.FP_PRECISION, msg='x_grad')

    @parameterized.expand([
        ('add:0', '__add__', -1.0, 1.0),
        ('add:1', '__add__', -1.0, 0.0),
        ('add:2', '__add__', 11.49, -2.321),

        ('sub:0', '__sub__', -1.0, 1.0),
        ('sub:1', '__sub__', -1.0, 0.0),
        ('sub:2', '__sub__', 11.49, -2.321),
        ('sub:3', '__sub__', -11.49, -2.321),
        
        ('mul:0', '__mul__', 0.0, 1.0),
        ('mul:1', '__mul__', 2.0, 3.0),
        ('mul:2', '__mul__', 28.42, -12.11),
        
        ('div:0', '__truediv__', 4.0, 2.0),
        ('div:1', '__truediv__', 1.0, 3.0),
        ('div:2', '__truediv__', 0.532, -1.2432),
        ('div:3', '__truediv__', 0.0, -1.2432),
    ])
    def test_binary_op(self, name, op, a, b):
        t_a = torch.tensor(a, requires_grad=True, dtype=torch.float64)
        t_b = torch.tensor(b, requires_grad=True, dtype=torch.float64)
        t_c = getattr(t_a, op)(t_b)
        t_c.backward()

        m_a = Node(a, 'a')
        m_b = Node(b, 'b')
        m_c = getattr(m_a, op)(m_b)
        m_c.backward()
        self.assertAlmostEqual(m_c.grad, 1.0, places=self.FP_PRECISION)
        self.assertAlmostEqual(t_c.item(), m_c.data, places=self.FP_PRECISION, msg='data')
        self.assertAlmostEqual(t_a.grad.item(), m_a.grad, places=self.FP_PRECISION, msg='a_grad')
        self.assertAlmostEqual(t_b.grad.item(), m_b.grad, places=self.FP_PRECISION, msg='b_grad')

    @parameterized.expand([
        ('add', '__add__', 2.0, 3.0),
        ('mul', '__mul__', 2.0, 0.0),
        ('div', '__truediv__', 2.0, 3.0),
    ])
    def test_binary_op_with_constant(self, name, op, a, c):
        t_a = torch.tensor(a, requires_grad=True, dtype=torch.float64)
        t_c = getattr(t_a, op)(c)
        t_c.backward()

        m_a = Node(a, 'a')
        m_c = getattr(m_a, op)(c)
        m_c.backward()
        self.assertAlmostEqual(m_c.grad, 1.0, places=self.FP_PRECISION)
        self.assertAlmostEqual(t_c.item(), m_c.data, places=self.FP_PRECISION, msg='data')
        self.assertAlmostEqual(t_a.grad.item(), m_a.grad, places=self.FP_PRECISION, msg='a_grad')

    @parameterized.expand([
        ('add', 2.0, 3.0),
        ('mul', 2.0, 0.0),
        ('div', 2.0, 0.0),
        ('sub', 2.0, 0.0),
    ])
    def test_rdunders(self, name, a, c):
        m_a = Node(a, 'a')
        if name == 'add':
            self.assertEqual((c + m_a).data, c + m_a.data)
        elif name == 'mul':
            self.assertEqual((c * m_a).data, c * m_a.data)
        elif name == 'div':
            self.assertEqual((c / m_a).data, c / m_a.data)
        elif name == 'sub':
            self.assertEqual((c - m_a).data, c - m_a.data)
        else:
            raise NotImplementedError