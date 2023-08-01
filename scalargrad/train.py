import logging
import random

from .nn import Softmax


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
        self.logger.info(f'Epoch: {epoch}/{self.num_epochs}    Loss: {loss:.4f}')

        if self.evaluator is None:
            return
        self.logger.info('evaluating...')
        softmax = Softmax()
        outputs = []
        targets = []
        for x, y in self._iter_batches(self.dataset_val):
            for xi, yi in zip(x, y):
                pred = softmax(self.model(xi))
                pred = [p.data for p in pred]
                outputs.append(pred)
                targets.append(yi)
        score = self.metric(outputs, targets)
        self.logger.info(f'score: {score}')
        self.optimizer.zero_grad()


class AccuracyMetric:

    def __call__(self, outputs, targets):
        assert len(outputs) == len(targets)
        scores = []
        for oi, ti in zip(outputs, targets):
            pi = max(oi)
            pi = oi.index(pi)
            score = int(int(pi) == int(ti))
            scores.append(score)
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
