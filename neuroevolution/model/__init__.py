import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F


class DNN(chainer.Chain):
    def __init__(self, n_action):
        super(DNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 30)
            self.l2 = L.Linear(None, n_action)

    def update(self):
        for p in self.params():
            p.data += np.random.randn(*p.shape) * 0.005

    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        with chainer.using_config('train', False):
            act = F.argmax(self(state))

        return int(act.data)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(x)
        return y


def create_model(n_action):
    return DNN(n_action)
