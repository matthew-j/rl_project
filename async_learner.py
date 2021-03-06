import torch
import torch.nn.functional as F

from  models import ActorCriticNN, QLearningNN
from shared_optimization import copy_learner_grads

def a3c_learner(pnum, target_model, Tlock, Tmax, T, max_steps, learner_policy, gamma, beta, optimizer, env_generator):
    env = env_generator.generate_env()
    torch.manual_seed(1 + pnum)
    model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    model.train()
    done = True
    
    while(T.data < Tmax):
        model.load_state_dict(target_model.state_dict())

        state_values = []
        log_probs = []
        rewards = []
        entropy = []

        if done:
            cur_state = env.reset()
            cur_state = torch.tensor([cur_state.__array__().tolist()])
            cell_state = torch.zeros(1, 256)
            hidden_state = torch.zeros(1, 256)
        else:
            cell_state = cell_state.detach()
            hidden_state = hidden_state.detach()

        for step in range(max_steps):
            q_values, state_value, (cell_state, hidden_state) = model((cur_state, (hidden_state, cell_state)))
            action_probs = F.softmax(q_values, dim=-1)
            action_log_probs = F.log_softmax(q_values, dim=-1)
            action = learner_policy.get_action(action_probs)

            next_state, reward, done, _ = env.step(action)

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
        if not done:
            q_values, state_value, (hidden_state, cell_state) = model((cur_state, (hidden_state, cell_state)))
            R = state_value.data

        q_loss = 0
        v_loss = 0
        state_values.append(R)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            q_loss = q_loss - log_probs[i] * (R - state_values[i]).data - beta * entropy[i]
            v_loss = v_loss + 0.5 * (R - state_values[i]).pow(2)
            
        optimizer.zero_grad()
        total_loss = q_loss + .5 * v_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 250)
        copy_learner_grads(model, target_model)
        optimizer.step()

def q_learner(pnum, target_model, behavioral_model, Tlock, Tmax, T, max_steps, learner_policy, gamma, I_target, optimizer, env_generator):
    env = env_generator.generate_env()
    torch.manual_seed(1 + pnum)

    done = True
    process_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    process_model.train()

    while(T.data < Tmax):
        process_model.load_state_dict(behavioral_model.state_dict())
        loss = torch.zeros(1, 1)

        if done:
            cur_state = env.reset()
            cur_state = torch.tensor([cur_state.__array__().tolist()])
            cell_state_process = torch.zeros(1, 256)
            hidden_state_process = torch.zeros(1, 256)
        else:
            cell_state_process = cell_state_process.detach()
            hidden_state_process = hidden_state_process.detach()
        
        for step in range(max_steps):
            q_values, (hidden_state_process, cell_state_process) = process_model(
                (cur_state, (hidden_state_process, cell_state_process))
            )
            action_probs = F.softmax(q_values, dim=-1)
            action = learner_policy.get_action(action_probs)

            next_state, reward, done, info = env.step(action)
            cur_state = torch.tensor([next_state.__array__().tolist()])

            q_value  = q_values.gather(-1, torch.tensor([[action]]))

            if not done:
                _, target_q_value, _ = target_model.act(
                    (cur_state, (hidden_state_process, cell_state_process)), epsilon=0
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
        process_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(process_model.parameters(), 250)
        copy_learner_grads(process_model, behavioral_model)
        optimizer.step()

def nstep_q_learner(pnum, target_model, behavioral_model, Tlock, Tmax, T, max_steps, learner_policy, gamma, I_target, optimizer, env_generator):
    env = env_generator.generate_env()
    torch.manual_seed(1 + pnum)
    
    done = True
    process_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    process_model.train()

    while(T.data < Tmax):
        process_model.load_state_dict(behavioral_model.state_dict())
        loss = torch.zeros(1, 1)

        if done:
            cur_state = env.reset()
            cur_state = torch.tensor([cur_state.__array__().tolist()])
            cell_state_process = torch.zeros(1, 256)
            hidden_state_process = torch.zeros(1, 256)
        else:
            cell_state_process = cell_state_process.detach()
            hidden_state_process = hidden_state_process.detach()

        rewards = []
        state_action_values = []

        for step in range(max_steps):
            q_values, (hidden_state_process, cell_state_process) = process_model(
                (cur_state, (hidden_state_process, cell_state_process))
            )
            action_probs = F.softmax(q_values, dim=-1)
            action = learner_policy.get_action(action_probs)

            next_state, reward, done, info = env.step(action)
            cur_state = torch.tensor([next_state.__array__().tolist()])
            rewards.append(reward)
            state_action_values.append(q_values.gather(-1, torch.tensor([[action]])))

            with Tlock:
                T += 1

            if T % I_target == 0:
                target_model.load_state_dict(behavioral_model.state_dict())

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            _, target_q_value, _ = target_model.act(
                (cur_state, (hidden_state_process, cell_state_process)), epsilon=0
            )
            R = target_q_value.data
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            loss = loss + 0.5 * (R - state_action_values[i]).pow(2)

        optimizer.zero_grad()
        process_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(process_model.parameters(), 250)
        copy_learner_grads(process_model, behavioral_model)
        optimizer.step()
