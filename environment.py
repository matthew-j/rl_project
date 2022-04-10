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

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

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

def generate_env(actions= [
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ], skip_num=4, frame_stack=4):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, actions)
    env = GrayScale(env)
    env = ResizeObservation(env, shape=84)
    env = NormalizedEnv(env)
    env = CustomReward(env)
    env = FrameStack(env, num_stack=frame_stack)
    env = SkipFrames(env, skip=skip_num)

    return env