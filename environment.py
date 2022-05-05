import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import torch
from torchvision import transforms
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

## Create wrappers for environment. Adapted from: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
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

# normalization technique inspired by:
# https://github.com/sadeqa/Super-Mario-Bros-RL/blob/master/A3C/common/atari_wrapper.py#L153
class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        if observation is not None:    # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)

        else:
            return observation

# modification to the gym_super_mario_bros reward function inspired by:
# https://github.com/sadeqa/Super-Mario-Bros-RL/blob/master/A3C/common/atari_wrapper.py#L76
class CustomReward(gym.Wrapper):
    def __init__(self, env=None):
        gym.Wrapper.__init__(self, env)
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.counter = 0

    def step(self, action):
        ''' 
            Implementing custom rewards
                Time = -0.1
                Distance = +1 or 0 
                Player Status = +/- 5
                Score = 2.5 x [Increase in Score]
                Done = +50 [Game Completed] or -50 [Game Incomplete]
        '''
        obs, true_reward, done, info = self.env.step(action)
            
        reward = max(min((info['x_pos'] - self.prev_dist - 0.05), 2), -2)
        self.prev_dist = info['x_pos']
        
        reward += (self.prev_time - info['time']) * -0.1
        self.prev_time = info['time']

        reward += (int(info['status']!='small')  - self.prev_stat) * 5
        self.prev_stat = int(info['status']!='small')

        reward += (info['score'] - self.prev_score) * 0.025
        self.prev_score = info['score']

        if done:
            if info['flag_get'] :
                reward += 500
            else:
                reward -= 50
        
        return obs, reward/10, done, info

    def reset(self):
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.counter = 0
        return self.env.reset()

    def change_level(self, level):
        self.env.change_level(level)

"""Static action sets for binary to discrete action space wrappers."""
# Ref: https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
# actions for the simple run right environment
action_spaces = {
    "easy_movement": [
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ],
    "right_only": [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ],
    # actions for very simple movement
    "simple_movement": [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
    ],
    # actions for more complex movement
    "complex_movement": [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ]
}

class CustomEnvironment:
    def __init__(self, env_name, actions, skip_num, frame_stack):
        self.actions = actions
        self.skip_num = skip_num
        self.frame_stack = frame_stack
        self.env_name = env_name

    def generate_env(self):
        env = gym_super_mario_bros.make(self.env_name)
        env = JoypadSpace(env, self.actions)
        env = GrayScale(env)
        env = ResizeObservation(env, shape=84)
        env = NormalizedEnv(env)
        env = CustomReward(env)
        env = FrameStack(env, num_stack=self.frame_stack)
        env = SkipFrames(env, skip=self.skip_num)

        return env