import random
import unittest

from parameterized import parameterized

try:
    import torch
except ImportError:
    raise ImportError('PyTorch required to run tests')

from minigrad.nn import Neuron


class TestNN(unittest.TestCase):
    FP_PRECISION = 5

    @parameterized.expand([
        (2, 'identity'),
        (2, 'relu'),
        (2, 'sigmoid'),

        (1, 'identity'),
        (1, 'relu'),
        (1, 'sigmoid'),

        (10, 'identity'),
        (10, 'relu'),
        (10, 'sigmoid'),

        (100, 'identity'),
        (100, 'relu'),
        (100, 'sigmoid'),

        (500, 'identity'),
        (500, 'relu'),
        (500, 'sigmoid'),
        
    ])
    def test_neuron(self, size, activation):
        inp = [random.uniform(-1.0, 1.0) for _ in range(size)]
        w = [random.uniform(-1.0, 1.0) for _ in range(size)]
        b = random.uniform(-1.0, 1.0)

        t_inp = torch.tensor(inp, dtype=torch.float64, requires_grad=False)
        t_w = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        t_b = torch.tensor(b, dtype=torch.float64, requires_grad=True)
        t_out = (t_inp * t_w).sum() + t_b
        t_out = t_out.sum()
        if activation != 'identity':
            t_out = getattr(t_out, activation)()
        t_out.retain_grad()
        t_out.backward()
        t_grad = [*t_w.grad.tolist(), t_b.grad.item()]

        m_neuron = Neuron(size=size, activation=activation)
        m_neuron.initialize(w=w, b=b)
        m_out = m_neuron(inp)
        m_out.backward()
        m_grad = [p.grad for p in m_neuron.parameters()]

        self.assertAlmostEqual(t_out.item(), m_out.data, places=self.FP_PRECISION, msg='data')
        self.assertEqual(len(t_grad), len(m_grad), msg='size')
        for t_v, m_v in zip(t_grad, m_grad):
            self.assertAlmostEqual(t_v, m_v, places=self.FP_PRECISION, msg='grad')



