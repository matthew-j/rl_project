import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import torch
from torchvision import transforms
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

## Create wrappers for environment. Inspired by: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.num_skips = skip
    
    def step(self, action):
        tot_reward = 0
        done = False
        for _ in range(self.num_skips):
            state, reward, done, info = self.env.step(action)
            tot_reward += reward
            if done:
                break
        return state, tot_reward, done, info 

class GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        func = transforms.Grayscale()
        observation = func(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        func = transforms.Compose(
            [transforms.Resize(self.shape), transforms.Normalize(0, 255)]
        )
        observation = func(observation).squeeze(0)

        return observation

def generate_env(actions, skip_num, frame_stack):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, actions)
    env.reset()
    env = SkipFrames(env, skip=skip_num)
    env = GrayScale(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=frame_stack)

    return env