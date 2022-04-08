from concurrent.futures import process
import torch
import torch.multiprocessing as mp
from torch import argmax
from a3c_evaluator import evaluate
from a3c_learner import train
from models import ActorCriticNN
from environment import generate_env
from torch.distributions import Categorical

def sample_action(action_values):
    dist = Categorical(action_values)
    return dist.sample().item()

def argmax_action(action_values):
    return argmax(action_values).item()

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def main():
    num_processes = 8

    env = generate_env()
    target_model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    target_model.share_memory()
    optimizer = SharedAdam(target_model.parameters(), lr = 0.0001)
    
    processes = []
    p = mp.Process(target = evaluate, args = (target_model, 10000, False))
    processes.append(p)
    p.start()

    for i in range(0, num_processes):
        if i < num_processes // 2:
            p = mp.Process(target = train, args = (i, target_model, 10000, 50, sample_action, .9, .01, optimizer))
        else:
            p = mp.Process(target = train, args = (i, target_model, 10000, 50, argmax_action, .9, .01, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
