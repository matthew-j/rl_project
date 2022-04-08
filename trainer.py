import argparse
import torch
from torch import argmax
import torch.multiprocessing as mp
from torch.distributions import Categorical

from evaluator import evaluate
from async_learner import a3c_learner, q_learner, nstep_q_learner
from models import ActorCriticNN, QLearningNN
from environment import generate_env
from shared_optimization import SharedAdam

## Argparse
parser = argparse.ArgumentParser(description='Train mario.')
parser.add_argument('algorithm', metavar='a', type=str, 
                    help='algorithm to train with', default=None)
parser.add_argument('--processes', metavar='p', type=int, 
                    help='number of processes to train with', default=4)
parser.add_argument('--tmax', metavar='t', type=int, 
                    help='number of steps to run', default=10000000)
parser.add_argument('--render', metavar='r', type=bool, 
                    help='whether to render mario', default=False)
parser.add_argument('--model_file', metavar='m', type=str, 
                    help='model file to train with', default=None)

## Functions for learners to pick action
def sample_action(action_values):
    dist = Categorical(action_values)
    return dist.sample().item()

def argmax_action(action_values):
    return argmax(action_values).item()

def train_a3c(num_processes, Tmax, render, model_file):
    env = generate_env()
    target_model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    target_model.share_memory()

    if model_file is not None:
        target_model.load_state_dict(model_file)

    optimizer = SharedAdam(target_model.parameters(), lr = 0.0001)

    T = torch.tensor(0)
    T.share_memory_()
    Tlock = mp.Lock()
    
    processes = []
    p = mp.Process(target = evaluate, 
        args = ("a3c", target_model, ActorCriticNN(env.observation_space.shape, env.action_space.n), 
        T, Tmax, render)
    )
    processes.append(p)
    p.start()

    for i in range(0, num_processes):
        if i < num_processes // 2:
            p = mp.Process(target = a3c_learner, args = (i, target_model, Tlock, Tmax, T, 50, sample_action, .9, .01, optimizer))
        else:
            p = mp.Process(target = a3c_learner, args = (i, target_model, Tlock, Tmax, T, 50, argmax_action, .9, .01, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def train_qlearning(num_processes, Tmax, render, model_file):
    env = generate_env()
    target_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    behavioral_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    target_model.share_memory()
    behavioral_model.share_memory()

    if model_file is not None:
        target_model.load_state_dict(model_file)

    optimizer = SharedAdam(behavioral_model.parameters(), lr = 0.0001)

    T = torch.tensor(0)
    T.share_memory_()
    Tlock = mp.Lock()
    
    processes = []
    p = mp.Process(target = evaluate, 
        args = ("qlearn", target_model, QLearningNN(env.observation_space.shape, env.action_space.n), 
        T, Tmax, render)
    )
    processes.append(p)
    p.start()

    for i in range(0, num_processes):
        p = mp.Process(target=q_learner, args=(i, target_model, behavioral_model, Tlock, Tmax, T, 50, 1, 0.99, 0.9, 500, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def train_qlearning(num_processes, Tmax, render, model_file):
    env = generate_env()
    target_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    behavioral_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    target_model.share_memory()
    behavioral_model.share_memory()

    if model_file is not None:
        target_model.load_state_dict(model_file)

    optimizer = SharedAdam(behavioral_model.parameters(), lr = 0.0001)

    T = torch.tensor(0)
    T.share_memory_()
    Tlock = mp.Lock()
    
    processes = []
    p = mp.Process(target = evaluate, 
        args = ("qlearn", target_model, QLearningNN(env.observation_space.shape, env.action_space.n), 
        T, Tmax, render)
    )
    processes.append(p)
    p.start()

    for i in range(0, num_processes):
        p = mp.Process(target=q_learner, args=(i, target_model, behavioral_model, Tlock, Tmax, T, 50, 1, 0.99, 0.9, 500, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def train_nstep_qlearning(num_processes, Tmax, render, model_file):
    env = generate_env()
    target_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    behavioral_model = QLearningNN(env.observation_space.shape, env.action_space.n)
    target_model.share_memory()
    behavioral_model.share_memory()

    if model_file is not None:
        target_model.load_state_dict(model_file)

    optimizer = SharedAdam(behavioral_model.parameters(), lr = 0.0001)

    T = torch.tensor(0)
    T.share_memory_()
    Tlock = mp.Lock()
    
    processes = []
    p = mp.Process(target = evaluate, 
        args = ("nqlearn", target_model, QLearningNN(env.observation_space.shape, env.action_space.n), 
        T, Tmax, render)
    )
    processes.append(p)
    p.start()

    for i in range(0, num_processes):
        p = mp.Process(target=nstep_q_learner, args=(i, target_model, behavioral_model, Tlock, Tmax, T, 50, 1, 0.99, 0.9, 500, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    args = parser.parse_args()

    num_processes = args.processes
    model_file = args.model_file
    render = args.render
    Tmax = args.tmax

    if args.algorithm == "a3c":
        train_a3c(num_processes, Tmax, render, model_file)
    elif args.algorithm == "qlearn":
        train_qlearning(num_processes, Tmax, render, model_file)
    elif args.algorithm == "nqlearn":
        train_nstep_qlearning(num_processes, Tmax, render, model_file)
    else:
        print("Wrong algorithm")
