import logging
import random


class Trainer:

    def __init__(
        self,
        x,
        y,
        model,
        loss,
        optimizer,
        batch_size,
        num_epochs,
        evaluator=None,
        logger=None,
    ):
        self.x = x
        self.y = y
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluator = evaluator
        self.logger = logger
        self.losses = None
        if self.logger is None:
            fmt = '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
            logger = logging.getLogger('scalargrad')
            logger.setLevel('INFO')
            ch = logging.StreamHandler()
            ch.setLevel('INFO')
            formatter = logging.Formatter(fmt)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            self.logger = logger
    
    def run(self):
        self.logger.info('Training...')
        self.losses = []
        batch_size = self.batch_size
        for epoch in range(self.num_epochs):
            epoch += 1
            for i in range(0, len(self.x), batch_size):
                xb = self.x[i:i + batch_size]
                yb = self.y[i:i + batch_size]
                outputs = [self.model(xi) for xi in xb]
                targets = yb
                loss = self.loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.data)
            self._on_epoch_end(epoch)
    
    def _on_epoch_end(self, epoch):
        loss = sum(self.losses) / len(self.losses)
        
        if self.evaluator is not None:
            score = self.evaluator.run()
            self.logger.info(f'Epoch: {epoch}/{self.num_epochs}    Loss: {loss:.8f}    {self.evaluator.metric.NAME}: {score:.4f}')
        else:
            self.logger.info(f'Epoch: {epoch}/{self.num_epochs}    Loss: {loss:.8f}')


class Evaluator:

    def __init__(
        self,
        x,
        y,
        model,
        metric,
    ):
        self.x = x
        self.y = y
        self.model = model
        self.metric = metric
        self.logger = None
    
    def run(self):
        outputs = [self.model(xi) for xi in self.x]
        targets = self.y
        score = self.metric(outputs, targets)
        return score


class AccuracyMetric:
    NAME = 'Accuracy'

    def __call__(self, outputs, targets):
        assert len(outputs) == len(targets)
        scores = []
        for oi, ti in zip(outputs, targets):
            oi = [n.data for n in oi]
            pi = max(oi)
            pi = oi.index(pi)
            score = int(int(pi) == int(ti))
            scores.append(score)
        score = sum(scores) / len(scores)
        return score


class MSEMetric:
    NAME = 'MSE'

    def __call__(self, outputs, targets):
        assert len(outputs) == len(targets)
        assert set(len(o) for o in outputs) == {1}
        scores = [((oi[0].data - ti) ** 2) for oi, ti in zip(outputs, targets)]
        score = sum(scores) / len(scores)
        return score


def split_train_val(x, y, val_size):
    val_size = int(round(len(x) * val_size, 0)) if isinstance(val_size, float) else val_size
    val_ixs = set(random.sample(range(len(x)), val_size))
    x_train = [xi for i, xi in enumerate(x) if i not in val_ixs]
    x_val = [xi for i, xi in enumerate(x) if i in val_ixs]
    y_train = [yi for i, yi in enumerate(y) if i not in val_ixs]
    y_val = [yi for i, yi in enumerate(y) if i in val_ixs]
    return x_train, y_train, x_val, y_val
