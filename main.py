import gym
from PIL import Image


def main():
    env = gym.make('CartPole-v0')
    env.reset()
    img = Image.fromarray(env.render(mode='rgb_array'))
    img.save('./hoge.png')
    print(img)


if __name__ == '__main__':
    main()
