import gym


def create_environment():
    env = gym.make('CartPole-v0')
    return env
