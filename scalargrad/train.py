import logging
import random

from .nn import Softmax


def split_train_val(x, y, val_size):
    val_size = int(round(len(x) * val_size, 0)) if isinstance(val_size, float) else val_size
    val_ixs = set(random.sample(range(len(x)), val_size))
    x_train = [xi for i, xi in enumerate(x) if i not in val_ixs]
    x_val = [xi for i, xi in enumerate(x) if i in val_ixs]
    y_train = [yi for i, yi in enumerate(y) if i not in val_ixs]
    y_val = [yi for i, yi in enumerate(y) if i in val_ixs]
    return x_train, y_train, x_val, y_val


class Dataset:

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class XYDataset(Dataset):

    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


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


class Trainer:

    def __init__(
        self,
        dataset_train,
        model,
        loss,
        optimizer,
        batch_size,
        num_epochs,
        metric=None,
        dataset_val=None,
        logger=None,
    ):
        self.dataset_train = dataset_train
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.metric = metric
        self.dataset_val = dataset_val
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger('scalargrad')
            self.logger.setLevel('INFO')
            logging.basicConfig()
    
    def run(self):
        losses = []
        step = 0
        for epoch in range(self.num_epochs):
            epoch += 1
            for x, y in self._iter_batches(self.dataset_train):
                step += 1
                outputs = [self.model(xi) for xi in x]
                targets = y
                loss = self.loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data)
                loss_value = sum(losses) / len(losses)
                self.logger.info(f'[train step] epoch: {epoch} step: {step} loss: {loss_value:.4f}')
            self._on_epoch_end()
    
    def _iter_batches(self, dataset):
        step = self.batch_size
        for i in range(0, len(dataset), step):
            batch = [dataset[j] for j in range(i, i + step) if j < len(dataset)]
            x = [it[0] for it in batch]
            y = [it[1] for it in batch]
            yield x, y

    def _on_epoch_end(self):
        if self.metric is None or self.dataset_val is None:
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
