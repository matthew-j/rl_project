import logging
from environment import generate_env
from agent import Agent
from models import  *
from collections import deque
import numpy as np
import torch
import cv2

def calculate_return(reward_queue, gamma):
    ret = reward_queue.popleft()
    for i in range(len(reward_queue)):
        ret += reward_queue[i] * gamma**(i + 1)
    return ret

## less resizing
## bigger model
## q learning
## v3.0

def train():
    ## Train Parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    render = False
    load_model = False
    model_file = "saves/MiniCnn3000.pt"

    model_save_freq = 500
    batch_size = 64
    logging_freq = 100
    cumulative_rewards = []

    ## Agent / environment Parameters
    joystick_actions = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
    ]
    num_actions = len(joystick_actions)
    frame_skips = 8
    frame_stack = 4
    alpha = 0.0001
    epsilon = 0.1

    # Sarsa Parameters
    MAX_STEPS = int(5000 / frame_skips)
    num_episodes = 50000
    n = 4
    gamma = .9

    model = Cnn((frame_stack, 84, 84), num_actions)
    env = generate_env(joystick_actions, frame_skips, frame_stack)
    agent = Agent(alpha, model, epsilon, num_actions, batch_size, device)

    if load_model:
        agent.load_model(model_file)

    memory = []
    total_steps = 0
    for episode in range(num_episodes):
        cumulative_reward = 0
        # Sarsa Objects
        reward_queue = deque(maxlen=n)
        action_queue = deque(maxlen=n)
        state_queue = deque(maxlen=n)

        done = False
        cur_state = env.reset()
        cur_state = torch.tensor([np.array(cur_state)])
        state_queue.append(cur_state)
        start_action = agent.get_action(cur_state)
        action_queue.append(start_action)

        T = np.inf

        for t in range(MAX_STEPS):
            if t < T:
                action = action_queue[-1]

                state, reward, done, info = env.step(action)
                cumulative_reward += reward
                if render:
                    env.render()
                cur_state = torch.tensor([np.array(state)])
                next_action = agent.get_action(cur_state)
                action_queue.append(next_action)
                state_queue.append(cur_state)
                reward_queue.append(reward)
                if done:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = calculate_return(reward_queue, gamma)
                if tau + n < T:
                    with torch.no_grad():
                        G += (gamma**n) * agent(state_queue[-1], action_queue[-1]).cpu().item()
                memory.append((torch.tensor([G]), state_queue[0], torch.tensor([action_queue[0]])))
                total_steps += 1
                if total_steps % batch_size == 0:
                    Gs, states, actions = map(torch.cat, zip(*memory))
                    agent.update(Gs, states, actions)
                    memory.clear()

            if tau == T - 1:
                break
        
        cumulative_rewards.append(cumulative_reward)
        agent.update_epsilon()

        ## Logging
        if episode % logging_freq == 0:
            print("Episode {} out of {}. Avg reward: {}. Eps = {}"
                .format(episode, num_episodes, sum(cumulative_rewards[-logging_freq:]) / logging_freq, agent.get_epsilon())
            )

        ## Model Saving
        if episode % model_save_freq == 0:
            agent.save_model("saves/MiniCnn{}.pt".format(episode))

    env.close()

if __name__ == "__main__":
    train()
