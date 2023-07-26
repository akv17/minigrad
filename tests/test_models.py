import unittest
import random

from parameterized import parameterized

try:
    import torch
except ImportError:
    raise ImportError('PyTorch required to run tests')

from scalargrad.nn import Linear, Sequential
from tests.util import check_arr


class TestModels(unittest.TestCase):

    @parameterized.expand([
        (4, 2, [8, 16, 32], 'relu', 'identity'),
        (4, 2, [8, 16, 32], 'sigmoid', 'identity'),
        
        (64, 32, [8, 16, 32], 'relu', 'identity'),
        (64, 32, [8, 16, 32], 'sigmoid', 'identity'),

        (128, 64, [8, 16, 32, 64, 128], 'relu', 'identity'),
        (128, 64, [8, 16, 32, 64, 128], 'sigmoid', 'identity'),
    ])
    def test_mlp(self, size_in, size_out, layers, hidden_activation, out_activation):
        sizes_in = [size_in, *layers]
        sizes_out = [*layers, size_out]
        activations = [hidden_activation] * len(layers) + [out_activation]
        t_layers = []
        m_layers = []
        for s_in, s_out, act in zip(sizes_in, sizes_out, activations):
            if act == 'relu':
                t_act = torch.nn.ReLU().double()
            elif act == 'sigmoid':
                t_act = torch.nn.Sigmoid().double()
            elif act == 'identity':
                t_act = torch.nn.Identity().double()
            t_lin = torch.nn.Linear(s_in, s_out).double()
            t_layers.append(t_lin)
            t_layers.append(t_act)
            m_lin = Linear.from_torch(t_lin, activation=act)
            m_layers.append(m_lin)
        t_model = torch.nn.Sequential(*t_layers)
        m_model = Sequential(m_layers, do_initialize=False)

        inp = [random.uniform(-1.0, 1.0) for _ in range(size_in)]
        t_inp = torch.tensor(inp).double()
        t_out = t_model(t_inp)
        t_out = t_out.sum()
        t_out.backward()
        
        m_inp = inp
        m_out = m_model(m_inp)
        m_out = sum(m_out)
        m_out.backward()

        self.assertAlmostEqual(t_out.item(), m_out.data, places=4, msg='forward')
        
        t_layers = t_model[0::2]  # grab linear layers.
        m_layers = m_model.modules
        self.assertEqual(len(t_layers), len(m_layers))
        for t_layer, m_layer in zip(t_layers, m_layers):
            t_w_bwd = t_layer.weight.grad
            t_b_bwd = t_layer.bias.grad
            m_neurons = m_layer._neurons
            self.assertEqual(len(t_w_bwd), len(m_neurons))
            self.assertEqual(len(t_b_bwd), len(m_neurons))
            for cur_t_w_bwd, cur_t_b_bwd, cur_neuron in zip(t_w_bwd, t_b_bwd, m_neurons):
                cur_t_w_bwd = cur_t_w_bwd.tolist()
                cur_t_b_bwd = [cur_t_b_bwd.item()]
                cur_neuron_w_bwd = [w.grad for w in cur_neuron._w]
                cur_neuron_b_bwd = [cur_neuron._b.grad]
                self.assertTrue(check_arr(cur_t_w_bwd, cur_neuron_w_bwd, tol=1e-4), msg='w_grad')
                self.assertTrue(check_arr(cur_t_b_bwd, cur_neuron_b_bwd, tol=1e-4), msg='b_grad')
