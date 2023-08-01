import os
import unittest
import pickle

from scalargrad.nn import Linear, CrossEntropyLoss, MSELoss, SGD, Sequential
from tests.util import check_arr, require_torch

torch = require_torch()


class TestTrain(unittest.TestCase):

    def test_classification(self):
        path = os.path.join('tests', 'data', 'clf.pkl')
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
        t_model = torch.nn.Sequential(
            torch.nn.Linear(16, 32).double(),
            torch.nn.ReLU().double(),
            torch.nn.Linear(32, 16).double(),
            torch.nn.Sigmoid().double(),
            torch.nn.Linear(16, 3).double(),
        ).double()
        m_model = Sequential([
            Linear.from_torch(t_model[0], activation='relu'),
            Linear.from_torch(t_model[2], activation='sigmoid'),
            Linear.from_torch(t_model[4], activation='identity'),

        ], do_initialize=False)
        t_loss = torch.nn.CrossEntropyLoss().double()
        m_loss = CrossEntropyLoss()
        self._test(
            x=x,
            y=y,
            t_model=t_model,
            m_model=m_model,
            t_loss=t_loss,
            m_loss=m_loss,
            batch_size=4,
            num_epochs=10
        )
    
    def test_regression(self):
        path = os.path.join('tests', 'data', 'reg.pkl')
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
        t_model = torch.nn.Sequential(
            torch.nn.Linear(16, 32).double(),
            torch.nn.ReLU().double(),
            torch.nn.Linear(32, 16).double(),
            torch.nn.Sigmoid().double(),
            torch.nn.Linear(16, 1).double(),
        ).double()
        m_model = Sequential([
            Linear.from_torch(t_model[0], activation='relu'),
            Linear.from_torch(t_model[2], activation='sigmoid'),
            Linear.from_torch(t_model[4], activation='identity'),

        ], do_initialize=False)
        t_loss = torch.nn.MSELoss().double()
        m_loss = MSELoss()
        self._test(
            x=x,
            y=y,
            t_model=t_model,
            m_model=m_model,
            t_loss=t_loss,
            m_loss=m_loss,
            batch_size=3,
            num_epochs=10,
            is_regession=True
        )

    def _test(
        self,
        x,
        y,
        t_model,
        m_model,
        t_loss,
        m_loss,
        batch_size,
        num_epochs,
        is_regession=False,
    ):
        t_x = torch.tensor(x).double()
        if is_regession:
            t_y = torch.tensor(y).double()
        else:
            t_y = torch.tensor(y).long()
        t_layers = list(t_model)[::2]
        m_layers = m_model.modules
        lr = 0.01
        t_optim = torch.optim.SGD(t_model.parameters(), lr=lr)
        m_optim = SGD(m_model.parameters(), lr=lr)
        t_losses = []
        m_losses = []
        step = 0
        total_steps = num_epochs * len(x) // batch_size
        print()
        for _ in range(num_epochs):
            for i in range(0, len(x), batch_size):
                step += 1
                xb = x[i:i+batch_size]
                yb = y[i:i+batch_size]
                t_xb = t_x[i:i+batch_size].double()
                if is_regession:
                    t_yb = t_y[i:i+batch_size].double()
                else:
                    t_yb = t_y[i:i+batch_size].long()

                t_optim.zero_grad()
                t_out = t_model(t_xb)
                if is_regession:
                    t_out = t_out.view(-1)
                t_loss_val = t_loss(t_out, t_yb)
                t_loss_val.backward()
                t_optim.step()
                
                m_optim.zero_grad()
                m_out = [m_model(xi) for xi in xb]
                m_loss_val = m_loss(m_out, yb)
                m_loss_val.backward()
                m_optim.step()

                self.assertAlmostEqual(t_loss_val.item(), m_loss_val.data, places=4, msg='loss')

                t_losses.append(t_loss_val.item())
                m_losses.append(m_loss_val.data)
                t_loss_running = sum(t_losses) / len(t_losses)
                m_loss_running = sum(m_losses) / len(m_losses)
                print(f'step: {step}/{total_steps}  torch_loss: {t_loss_running:.5f} scalargrad_loss: {m_loss_running:.5f}')

                for j, (t_layer, m_layer) in enumerate(zip(t_layers, m_layers)):
                    for k in range(len(m_layer._neurons)):
                        t_w_fwd = t_layer.weight.data[k].tolist()
                        t_w_bwd = t_layer.weight.grad[k].tolist()
                        t_b_fwd = t_layer.bias.data[k].item()
                        t_b_bwd = t_layer.bias.grad[k].item()
                        m_w_fwd = [w.data for w in m_layer._neurons[k]._w]
                        m_w_bwd = [w.grad for w in m_layer._neurons[k]._w]
                        m_b_fwd = m_layer._neurons[k]._b.data
                        m_b_bwd = m_layer._neurons[k]._b.grad
                        
                        self.assertTrue(check_arr(t_w_fwd, m_w_fwd, 1e-4, show_diff=True), msg=f'{j}_{k}::w_forward')
                        self.assertTrue(check_arr([t_b_fwd], [m_b_fwd], 1e-4, show_diff=True), msg=f'{j}_{k}::b_forward')
                        self.assertTrue(check_arr(t_w_bwd, m_w_bwd, 1e-4, show_diff=True), msg=f'{j}_{k}::w_backward')
                        self.assertTrue(check_arr([t_b_bwd], [m_b_bwd], 1e-4, show_diff=True), msg=f'{j}_{k}::b_backward')
