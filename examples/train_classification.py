import os
import pickle

from scalargrad.nn import Linear, Sequential, CrossEntropyLoss, SGD
from scalargrad.train import Trainer, Evaluator, AccuracyMetric, split_train_val

NUM_FEATURES = 16
NUM_CLASSES = 3
NUM_EPOCHS = 50
BATCH_SIZE = 4
LR = 0.01
MOMENTUM = 0.9


def main():
    x, y = _load_data()
    x_train, y_train, x_val, y_val = split_train_val(x, y, val_size=0.1)
    model = _build_model()
    loss = CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    evaluator = Evaluator(
        x=x_val,
        y=y_val,
        model=model,
        metric=AccuracyMetric(),
    )
    trainer = Trainer(
        x=x_train,
        y=y_train,
        model=model,
        loss=loss,
        optimizer=optim,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        evaluator=evaluator,
    )
    trainer.run()


def _load_data():
    path = os.path.join('tests', 'data', 'clf.pkl')
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
    return x, y


def _build_model():
    model = Sequential([
        Linear(NUM_FEATURES, 16, activation='relu'),
        Linear(16, 8, activation='relu'),
        Linear(8, NUM_CLASSES, activation='identity'),
    ])
    return model




if __name__ == '__main__':
    main()
