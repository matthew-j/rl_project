import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from  models import ActorCriticNN
from environment import generate_env

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(pnum, target_model, agent_episodes, max_steps, action_func, gamma, beta, optimizer):
    env = generate_env()
    torch.manual_seed(1 + pnum)
    model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    model.train()
    
    done = True
    for i in range(agent_episodes):
        cur_state = env.reset()
        cur_state = torch.tensor([cur_state.__array__().tolist()])
        model.load_state_dict(target_model.state_dict())

        if done:
            cell_state = torch.zeros(1, 256)
            hidden_state = torch.zeros(1, 256)
        else:
            cell_state = cell_state.detach()
            hidden_state = hidden_state.detach()

        state_values = []
        log_probs = []
        rewards = []
        entropy = []

        for step in range(max_steps):
            q_values, state_value, (cell_state, hidden_state) = model((cur_state, (hidden_state, cell_state)))

            action = action_func(q_values)
            dist = Categorical(q_values)

            next_state, reward, done, info = env.step(action)
            reward = max(min(reward, 50), -5)

            rewards.append(reward) 
            state_values.append(state_value)
            entropy.append(dist.entropy())
            log_probs.append(torch.log(q_values)[0][action])

            cur_state = torch.tensor([next_state.__array__().tolist()])

            if done:
                break
        
        R = torch.zeros(1,1)
        advantage_estimation = torch.zeros(1,1)
        if not done:
            q_values, state_value, (cell_state, hidden_state) = model((cur_state, (hidden_state, cell_state)))
            R += state_value.detach()

        q_loss = 0
        v_loss = 0
        state_values.append(R)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            advantage_estimation = advantage_estimation * gamma + (rewards[i] + gamma * state_values[i+1].detach() - state_values[i].detach())
            q_loss -= log_probs[i] * advantage_estimation - beta * entropy[i]
            v_loss += 0.5 * (R - state_values[i])**2
            
        optimizer.zero_grad()
        (q_loss + .5 * v_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 250)

        ensure_shared_grads(model, target_model)
        optimizer.step()




    