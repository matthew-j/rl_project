from environment import generate_env
from agent import Agent
from models import  MiniCnn
from collections import deque
import torch

def calculate_return(reward_queue, gamma, tau):
    ret = 0
    for i in range(len(reward_queue)):
        ret += reward_queue[i] * gamma**(i - tau - 1)
    return ret


def train():
    # Sarsa Parameters
    MAX_STEPS = 5000
    num_episodes = 10
    n = 5
    gamma = .9

    ## Agent / environment Parameters
    joystick_actions = [["right"], ["right", "A"]]
    num_actions = len(joystick_actions)
    frame_skips = 4
    frame_stack = 4
    alpha = 0.001
    epsilon = 0.1

    model = MiniCnn((frame_stack, 84, 84), num_actions)
    env = generate_env(joystick_actions, frame_skips, frame_stack)
    agent = Agent(alpha, model, epsilon, num_actions)

    for episode in range(num_episodes):
        print("Episode {} out of {}.", episode, num_episodes)

        # Sarsa Objects
        reward_queue = deque(maxlen=n)
        action_queue = deque(maxlen=n)
        state_queue = deque(maxlen=n)

        done = False
        cur_state = env.reset()
        cur_state = torch.tensor(cur_state)
        state_queue.append(cur_state)
        start_action = agent.get_action(cur_state)
        action_queue.append(start_action)

        for t in range(MAX_STEPS / frame_skips):
            action = action_queue[-1]

            state, reward, done, info = env.step(action)
            cur_state = state
            
            next_action = agent.get_action(cur_state)
            action_queue.append(next_action)
            state_queue.append(cur_state)
            reward_queue.append(reward)

            tau = t - n + 1
            if tau >= 0:
                G = calculate_return(reward_queue, gamma, tau)

                if tau + n < MAX_STEPS / frame_skips:
                    G += (gamma**n) * agent(cur_state, action_queue[-1])

                agent.update(G, state_queue[0], action_queue[0])

            if done:
                break
        
        

    env.close()

if __name__ == "__main__":
    train()
