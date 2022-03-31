import gym_super_mario_bros
from environment import generate_env
import cv2

env = generate_env()

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    # cv2.imshow('image', state[0].numpy())
    # cv2.waitKey(0)
    env.render()

env.close()
