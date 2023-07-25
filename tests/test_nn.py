import random
import unittest

from parameterized import parameterized

try:
    import torch
except ImportError:
    raise ImportError('PyTorch required to run tests')

from minigrad.nn import Neuron, Linear


class TestNN(unittest.TestCase):
    FP_PRECISION = 3

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

        self.assertAlmostEqual(t_out.item(), m_out.data, places=self.FP_PRECISION, msg='forward')

        t_bwd = self._round(t_grad)
        m_bwd = self._round(m_grad)
        self.assertEqual(len(t_grad), len(m_grad), msg='backward_len')
        self.assertListEqual(t_bwd, m_bwd, msg='backward_len')

    @parameterized.expand([
        (2, 1, 'identity'),
        (1, 5, 'identity'),
        (3, 4, 'identity'),
        (24, 32, 'identity'),

        (2, 1, 'relu'),
        (1, 5, 'relu'),
        (3, 4, 'relu'),
        (24, 32, 'relu'),

        (2, 1, 'sigmoid'),
        (1, 5, 'sigmoid'),
        (3, 4, 'sigmoid'),
        (24, 32, 'sigmoid'),


    ])
    def test_linear(self, size_in, size_out, activation):
        t_lin = torch.nn.Linear(size_in, size_out)

        w = t_lin.weight.data.tolist()
        b = t_lin.bias.data.tolist()
        inp = [random.uniform(-1, 1) for _ in range(size_in)]
        
        t_inp = torch.tensor(inp)
        t_out = t_lin(t_inp)
        if activation != 'identity':
            t_out = getattr(t_out, activation)()
        t_out = sum(t_out)
        # must reduce to scalar to enable backward.
        t_out.backward()
            
        
        m_neurons = [Neuron(size=size_in, activation=activation) for _ in range(size_out)]
        self.assertEqual(len(m_neurons), len(w))
        for m_n, wi, bi in zip(m_neurons, w, b):
            m_n.initialize(w=wi, b=bi)
        m_lin = Linear(size_in=size_in, size_out=size_out, activation=activation)
        m_lin.initialize(m_neurons)
        m_out = m_lin(inp)
        m_out = sum(m_out)
        m_out.backward()
        
        self.assertAlmostEqual(t_out.item(), m_out.data, places=self.FP_PRECISION, msg='forward')

        t_w_bwd = self._round(t_lin.weight.grad.ravel().tolist())
        m_w_bwd = self._round([w.grad for n in m_lin._neurons for w in n._w])
        self.assertEqual(len(t_w_bwd), len(m_w_bwd), msg='backward_len')
        self.assertListEqual(t_w_bwd, m_w_bwd, msg='backward')

    def _round(self, x):
        return [round(v, self.FP_PRECISION) for v in x]