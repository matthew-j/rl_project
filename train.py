import logging
from environment import generate_env
from agent import Agent
from models import  MiniCnn
from collections import deque
import numpy as np
import torch

def calculate_return(reward_queue, gamma, tau):
    ret = 0
    for i in range(len(reward_queue)):
        ret += reward_queue[i] * gamma**(i - tau - 1)
    return ret


def train():
    ## Train Parameters
    cuda = False
    render = False
    model_save_freq = 100
    logging_freq = 10

    ## Agent / environment Parameters
    joystick_actions = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ]
    num_actions = len(joystick_actions)
    frame_skips = 4
    frame_stack = 4
    alpha = 0.001
    epsilon = 0.1

    # Sarsa Parameters
    MAX_STEPS = int(5000 / frame_skips)
    num_episodes = 10000
    n = 8
    gamma = .9

    model = MiniCnn((frame_stack, 84, 84), num_actions)
    env = generate_env(joystick_actions, frame_skips, frame_stack)
    agent = Agent(alpha, model, epsilon, num_actions, cuda)

    for episode in range(num_episodes):
        cumulative_reward = 0
        # Sarsa Objects
        reward_queue = deque(maxlen=n)
        action_queue = deque(maxlen=n)
        state_queue = deque(maxlen=n)

        done = False
        cur_state = env.reset()
        cur_state = torch.tensor(np.array(cur_state))
        state_queue.append(cur_state)
        start_action = agent.get_action(cur_state)
        action_queue.append(start_action)

        for t in range(MAX_STEPS):
            action = action_queue[-1]

            state, reward, done, info = env.step(action)
            cumulative_reward += reward
            if render:
                env.render()
            cur_state = torch.tensor(np.array(state))
            
            next_action = agent.get_action(cur_state)
            action_queue.append(next_action)
            state_queue.append(cur_state)
            reward_queue.append(reward)

            tau = t - n + 1
            if tau >= 0:
                G = calculate_return(reward_queue, gamma, tau)

                if tau + n < MAX_STEPS:
                    G += (gamma**n) * agent(state_queue[-1], action_queue[-1])

                agent.update(G, state_queue[0], action_queue[0])

            if done:
                break
        
        ## Logging
        if episode % logging_freq == 0:
            print("Episode {} out of {}. Cumulative reward: {}. Eps = {}"
                .format(episode, num_episodes, cumulative_reward, agent.get_epsilon())
            )

        ## Model Saving
        if episode % model_save_freq == 0:
            agent.save_model("saves/MiniCnn{}.pt".format(episode))

    env.close()

if __name__ == "__main__":
    train()
