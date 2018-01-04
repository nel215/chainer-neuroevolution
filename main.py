import numpy as np
import gym
import chainer
import chainer.links as L
import chainer.functions as F
import random
from PIL import Image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class DNN(chainer.Chain):
    def __init__(self, n_action):
        super(DNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 5)
            self.l2 = L.Linear(None, n_action)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(x)
        return y


def get_action(state, dnn):
    state = np.array([state], dtype=np.float32)
    with chainer.using_config('train', False):
        act = F.argmax(dnn(state))

    return int(act.data)


def initialize_network(seed):
    set_random_seed(seed)
    env = gym.make('CartPole-v0')
    env.seed(seed)
    n_action = env.action_space.n
    dnn = DNN(n_action)
    state = env.reset()
    total_reward = 0
    while True:
        act = get_action(state, dnn)
        state, reward, done, info = env.step(act)
        total_reward += reward
        if done:
            break

    print(total_reward)
    return seed, total_reward


def main():
    initialize_network(215)
    env = gym.make('CartPole-v0')
    env.reset()
    img = Image.fromarray(env.render(mode='rgb_array'))
    img.save('./hoge.png')


if __name__ == '__main__':
    main()
