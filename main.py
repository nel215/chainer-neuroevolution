import os
import numpy as np
import gym
import random
from PIL import Image
from dask.distributed import Client
from neuroevolution import MinioModelStore
from neuroevolution.model import create_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_model_store():
    model_store = MinioModelStore(
        os.environ['MINIO_ACCESS_KEY'],
        os.environ['MINIO_SECRET_KEY'],
    )
    return model_store


def initialize_network(seed, store=False, name=None):
    set_random_seed(seed)
    env = gym.make('CartPole-v0')
    n_action = env.action_space.n
    agent = create_model(n_action)
    total_reward = run_episode(env, agent)

    if store:
        model_store = get_model_store()
        model_store.save(name, agent)

    return seed, total_reward


def run_episode(env, agent):
    env.seed(0)
    state = env.reset()
    total_reward = 0
    while True:
        act = agent.get_action(state)
        state, reward, done, info = env.step(act)
        total_reward += reward
        if done:
            break

    return total_reward


def update_network(seed, store=False, name=None):
    set_random_seed(seed)
    env = gym.make('CartPole-v0')
    n_action = env.action_space.n
    agent = create_model(n_action)
    rank = random.randint(0, 2)
    if seed == 0:
        pre_model_name = 'top-0'
    else:
        pre_model_name = 'top-{}'.format(rank)

    model_store = get_model_store()
    agent = model_store.load(pre_model_name, agent)
    if seed > 0:
        agent.update()

    total_reward = run_episode(env, agent)

    if store:
        model_store.save(name, agent)

    return seed, total_reward


def main():
    client = Client('scheduler:8786')
    futures = client.map(initialize_network, range(10))
    results = client.gather(futures)
    results.sort(key=lambda x: -x[1])

    truncated = list(map(lambda x: x[0], results[:3]))
    futures = []
    for i, seed in enumerate(truncated):
        name = 'top-{}'.format(i)
        futures.append(client.submit(
            initialize_network, seed, store=True, name=name))
    results = client.gather(futures)
    print(results, flush=True)

    for g in range(10):
        futures = client.map(update_network, range(10))
        results = client.gather(futures)
        results.sort(key=lambda x: -x[1])
        truncated = list(map(lambda x: x[0], results[:3]))

        futures = []
        for i, seed in enumerate(truncated):
            name = 'top-{}'.format(i)
            futures.append(client.submit(
                update_network, seed, store=True, name=name))
        results = client.gather(futures)
        print(results, flush=True)


if __name__ == '__main__':
    main()
