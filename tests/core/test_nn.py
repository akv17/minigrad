import random
import unittest

from parameterized import parameterized

from scalargrad.scalar import Scalar
from scalargrad.nn import Neuron, Linear, Softmax, CrossEntropyLoss, MSELoss, SGD
from tests.util import check_arr, require_torch

torch = require_torch()


class TestNN(unittest.TestCase):

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

        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')

        t_bwd = t_grad
        m_bwd = m_grad
        self.assertEqual(len(t_grad), len(m_grad), msg='backward_len')
        self.assertTrue(check_arr(t_bwd, m_bwd, tol=1e-4), msg='backward')

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
        t_out.backward()  # must reduce to scalar to enable backward.
        
        m_neurons = [Neuron(size=size_in, activation=activation) for _ in range(size_out)]
        self.assertEqual(len(m_neurons), len(w))
        for m_n, wi, bi in zip(m_neurons, w, b):
            m_n.initialize(w=wi, b=bi)
        m_lin = Linear(size_in=size_in, size_out=size_out, activation=activation)
        m_lin.initialize(m_neurons)
        m_out = m_lin(inp)
        m_out = sum(m_out)
        m_out.backward()
        
        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')

        t_w_bwd = t_lin.weight.grad.ravel().tolist()
        m_w_bwd = [w.grad for n in m_lin._neurons for w in n._w]
        self.assertEqual(len(t_w_bwd), len(m_w_bwd), msg='w_backward_len')
        self.assertTrue(check_arr(t_w_bwd, m_w_bwd, tol=1e-4), msg='w_backward')

        t_b_bwd = t_lin.bias.grad.ravel().tolist()
        m_b_bwd = [n._b.grad for n in m_lin._neurons]
        self.assertEqual(len(t_b_bwd), len(m_b_bwd), msg='b_backward_len')
        self.assertTrue(check_arr(t_b_bwd, m_b_bwd, tol=1e-4), msg='b_backward')

    @parameterized.expand([
        (1,),
        (2,),
        (8,),
        (32,),
        (128,),
        (1024,),
    ])
    def test_softmax(self, size):
        inp = torch.rand((size,), dtype=torch.float64).tolist()

        t_inp = torch.tensor(inp, dtype=torch.float64, requires_grad=True)
        t_out = t_inp.softmax(0).sum()
        t_out.backward()

        m_inp = [Scalar(v, name=f'inp{i}') for i, v in enumerate(inp)]
        m_out = Softmax()
        m_out = m_out(m_inp)
        m_out = sum(m_out)
        m_out.backward()

        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')
        t_bwd = t_inp.grad.ravel().tolist()
        m_bwd = [n.grad for n in m_inp]
        self.assertTrue(check_arr(t_bwd, m_bwd, tol=1e-4, show_diff=True))

    @parameterized.expand([
        (4, 1),
        (4, 2),
        (8, 4),
        (32, 16),
        (64, 128),
        (256, 128),
    ])
    def test_cross_entropy_loss(self, num_samples, num_classes):
        t_outputs = torch.rand((num_samples, num_classes), dtype=torch.float64, requires_grad=True)
        t_targets = torch.randint(0, num_classes, size=(num_samples,), dtype=torch.int64, requires_grad=False)

        outputs = t_outputs.tolist()
        targets = t_targets.tolist()

        t_out = torch.nn.CrossEntropyLoss()(t_outputs, t_targets)
        t_out.backward()

        m_outputs = [[Scalar(v, name=f'inp{i}_{j}') for j, v in enumerate(out)] for i, out in enumerate(outputs)]
        m_targets = targets
        m_out = CrossEntropyLoss()(m_outputs, m_targets)
        m_out.backward()

        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')
        t_bwd = [v.item() for v in t_outputs.grad.ravel()]
        m_bwd = [n.grad for out in m_outputs for n in out]
        self.assertTrue(check_arr(t_bwd, m_bwd, tol=1e-4, show_diff=True), msg='backward')

    @parameterized.expand([
        (4,),
        (8,),
        (32,),
        (128,),
        (256,),
        (512,),
        (1024,),
    ])
    def test_mse_loss(self, num_samples):
        t_outputs = torch.rand((num_samples,), dtype=torch.float64, requires_grad=True)
        t_targets = torch.rand((num_samples,), dtype=torch.float64, requires_grad=False)

        outputs = t_outputs.tolist()
        targets = t_targets.tolist()

        t_out = torch.nn.MSELoss()(t_outputs, t_targets)
        t_out.backward()

        m_outputs = [[Scalar(v, name=f'inp{i}')] for i, v in enumerate(outputs)]
        m_targets = targets
        m_out = MSELoss()(m_outputs, m_targets)
        m_out.backward()

        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')
        t_bwd = t_outputs.grad.ravel().tolist()
        m_bwd = [out[0].grad for out in m_outputs]
        self.assertTrue(check_arr(t_bwd, m_bwd, tol=1e-4, show_diff=True), msg='backward')

    @parameterized.expand([
        (4, 0.1, None, 1),
        (4, 0.1, None, 4),
        (4, 0.001, None, 1),
        (4, 0.001, None, 4),

        (4, 0.1, 0.9, 1),
        (4, 0.1, 0.9, 4),
        (4, 0.001, 0.9, 1),
        (4, 0.001, 0.9, 4),

        (512, 0.1, None, 1),
        (512, 0.1, None, 4),
        (512, 0.001, None, 1),
        (512, 0.001, None, 4),

        (512, 0.1, 0.9, 1),
        (512, 0.1, 0.9, 4),
        (512, 0.001, 0.9, 1),
        (512, 0.001, 0.9, 4),
    ])
    def test_sgd(self, num_params, lr, momentum, num_steps):
        params = torch.rand((num_params,), dtype=torch.float64).tolist()

        t_params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        t_sgd = torch.optim.SGD([t_params], lr=lr, momentum=momentum or 0.0)
        for _ in range(num_steps):
            t_sgd.zero_grad()
            t_out = t_params.exp().sum()
            t_out.backward()
            t_sgd.step()
        t_params_final = t_params.ravel().tolist()

        m_params = [Scalar(v) for v in params]
        m_sgd = SGD(parameters=m_params, lr=lr, momentum=momentum)
        for _ in range(num_steps):
            m_sgd.zero_grad()
            m_out = sum([p.exp() for p in m_params])
            m_out.backward()
            m_sgd.step()
        m_params_final = [p.data for p in m_params]

        self.assertEqual(len(t_params_final), len(m_params_final), msg='len')
        self.assertTrue(check_arr(t_params_final, m_params_final, tol=1e-4, show_diff=True), 'params')
