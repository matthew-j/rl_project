import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from  models import ActorCriticNN
from environment import generate_env
from shared_optimization import copy_learner_grads

def a3c_learner(pnum, target_model, Tlock, Tmax, T, max_steps, action_func, gamma, beta, optimizer):
    env = generate_env()
    torch.manual_seed(1 + pnum)
    model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    model.train()
    
    done = True
    while(T.data < Tmax):
        if done:
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
            action_probs = F.softmax(q_values, dim=-1)
            action_log_probs = F.log_softmax(q_values, dim=-1)
            action = action_func(action_probs)

            next_state, reward, done, _ = env.step(action)
            reward = max(min(reward, 50), -5)

            rewards.append(reward) 
            state_values.append(state_value)
            entropy.append(-(action_log_probs * action_probs).sum(-1, keepdim=True))
            log_probs.append(action_log_probs.gather(-1, torch.tensor([[action]])))

            cur_state = torch.tensor([next_state.__array__().tolist()])

            with Tlock:
                T += 1

            if done:
                break
        
        R = torch.zeros(1,1)
        advantage_estimation = torch.zeros(1,1)
        if not done:
            q_values, state_value, (hidden_state, cell_state) = model((cur_state, (hidden_state, cell_state)))
            R = state_value.data

        q_loss = 0
        v_loss = 0
        state_values.append(R)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            advantage_estimation = advantage_estimation * gamma + (rewards[i] + gamma * state_values[i+1].data - state_values[i].data)
            q_loss = q_loss - log_probs[i] * advantage_estimation - beta * entropy[i]
            v_loss = v_loss + 0.5 * (R - state_values[i]).pow(2)
            
        optimizer.zero_grad()
        total_loss = q_loss + .5 * v_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 250)

        copy_learner_grads(model, target_model)
        optimizer.step()

def q_learner(pnum, target_model, behavioral_model, Tlock, Tmax, T, max_steps, epsilon, epsilon_decay, gamma, I_target, lr):
    env = generate_env()
    torch.manual_seed(1 + pnum)

    done = True
    loss = torch.zeros(1, 1)
    optimizer = torch.optim.Adam(behavioral_model.parameters(), lr=lr)
    while(T.data < Tmax):
        if done:
            cur_state = env.reset()
            cur_state = torch.tensor([cur_state.__array__().tolist()])
            cell_state_behavior = torch.zeros(1, 256)
            hidden_state_behavior = torch.zeros(1, 256)
            cell_state_target = torch.zeros(1, 256)
            hidden_state_target = torch.zeros(1, 256)
        else:
            cell_state_behavior = cell_state_behavior.detach()
            hidden_state_behavior = hidden_state_behavior.detach()
            cell_state_target = cell_state_target.detach()
            hidden_state_target = hidden_state_target.detach()
        
        for step in range(max_steps):
            action, q_value, (hidden_state_behavior, cell_state_behavior) = behavioral_model.act(
                (cur_state, (hidden_state_behavior, cell_state_behavior)), epsilon
            )

            next_state, reward, done, info = env.step(action)
            cur_state = torch.tensor([next_state.__array__().tolist()])
            reward = max(min(reward, 50), -5)
            if not done:
                _, target_q_value, (hidden_state_target, cell_state_target) = target_model.act(
                    (cur_state, (hidden_state_target, cell_state_target)), epsilon=0
                )
                reward = reward + gamma * target_q_value.data

            loss = loss + .5 * (reward - q_value).pow(2)

            with Tlock:
                T += 1

            if T % I_target == 0:
                target_model.load_state_dict(behavioral_model.state_dict())

            if done:
                break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(behavioral_model.parameters(), 250)
        optimizer.step()
        loss = torch.zeros(1, 1)
        
        epsilon = max(epsilon * epsilon_decay, 0.1)

def nstep_q_learner(pnum, target_model, behavioral_model, Tlock, Tmax, T, max_steps, epsilon, epsilon_decay, gamma, I_target, lr):
    env = generate_env()
    torch.manual_seed(1 + pnum)
    
    done = True
    loss = torch.zeros(1, 1)
    optimizer = torch.optim.Adam(behavioral_model.parameters(), lr=lr)
    while(T.data < Tmax):
        if done:
            cur_state = env.reset()
            cur_state = torch.tensor([cur_state.__array__().tolist()])
            cell_state_behavior = torch.zeros(1, 256)
            hidden_state_behavior = torch.zeros(1, 256)
            cell_state_target = torch.zeros(1, 256)
            hidden_state_target = torch.zeros(1, 256)
        else:
            cell_state_behavior = cell_state_behavior.detach()
            hidden_state_behavior = hidden_state_behavior.detach()
            cell_state_target = cell_state_target.detach()
            hidden_state_target = hidden_state_target.detach()

        rewards = []
        state_action_values = []

        for step in range(max_steps):
            action, q_value, (hidden_state_behavior, cell_state_behavior) = behavioral_model.act(
                (cur_state, (hidden_state_behavior, cell_state_behavior)), epsilon
            )
            next_state, reward, done, info = env.step(action)
            cur_state = torch.tensor([next_state.__array__().tolist()])
            rewards.append(max(min(reward, 50), -5))
            state_action_values.append(q_value)

            with Tlock:
                T += 1

            if T % I_target == 0:
                target_model.load_state_dict(behavioral_model.state_dict())

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            _, target_q_value, (hidden_state_target, cell_state_target) = target_model.act(
                (cur_state, (hidden_state_target, cell_state_target)), epsilon=0
            )
            reward = reward + gamma * target_q_value.data
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            loss = loss + 0.5 * (R - q_value).pow(2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(behavioral_model.parameters(), 250)
        optimizer.step()
        loss = torch.zeros(1, 1)
        
        epsilon = max(epsilon * epsilon_decay, 0.1)
    