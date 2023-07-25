import unittest
from parameterized import parameterized

try:
    import torch
except ImportError:
    raise ImportError('PyTorch required to run tests')

from minigrad.node import Node


class TestOps(unittest.TestCase):
    FP_PRECISION = 5

    @parameterized.expand([
        ('add:0', '__add__', -1.0, 1.0),
        ('add:1', '__add__', -1.0, 0.0),
        ('add:2', '__add__', 11.49, -2.321),
        
        ('mul:0', '__mul__', 0.0, 1.0),
        ('mul:1', '__mul__', 2.0, 3.0),
        ('mul:2', '__mul__', 28.42, -12.11),
    ])
    def test_binary_op(self, name, op, a, b):
        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = getattr(t_a, op)(t_b)
        t_c.backward()

        m_a = Node(a, 'a')
        m_b = Node(b, 'b')
        m_c = getattr(m_a, op)(m_b)
        m_c.backward()
        self.assertAlmostEqual(m_c.data, t_c.item(), places=self.FP_PRECISION)
        self.assertAlmostEqual(m_c.grad, 1.0, places=self.FP_PRECISION)
        self.assertAlmostEqual(m_a.grad, t_a.grad.item(), places=self.FP_PRECISION)
        self.assertAlmostEqual(m_b.grad, t_b.grad.item(), places=self.FP_PRECISION)
