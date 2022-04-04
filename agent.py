import torch
from torch.distributions import Categorical
import numpy as np

class Agent:
    def __init__(self, alpha, model, epsilon, num_actions, batch_size, cuda):
        self.alpha = alpha
        self.model = model
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_function = torch.nn.SmoothL1Loss()

        self.epsilon = 1.000000
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1

        self.cuda = cuda
        if cuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

        self.batch_size = batch_size
    
    def get_epsilon(self):
        return self.epsilon

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        if self.cuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action_softmax(self, state):
        self.model.eval()
        if self.cuda:
            action_values = self.model(state.to(self.device))
        else:
            action_values = self.model(state)
        
        print("NN output:", action_values)
        action_value_dist = torch.nn.functional.softmax(action_values, dim=0)
        print("Softmax output: ", action_value_dist)
        prob_dist = Categorical(action_value_dist)
        action = prob_dist.sample().item()

        print("Action chosen:", action)
        return action 

    def get_action(self, state):
        self.model.eval()

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        else:
            if self.cuda:
                action_values = self.model(state.to(self.device))
            else:
                action_values = self.model(state)

            return torch.argmax(action_values, axis=1).item()

    def __call__(self, state, action):
        batches = state.size()[0]
        if self.cuda:
            return self.model(state.to(self.device))[
                np.arange(0, batches), action
            ]
        else:
            return self.model(state)[
                np.arange(0, batches), action
            ]

    def update(self, G, s_tau, a_tau):
        self.model.train()
        if self.cuda:
            loss = self.loss_function(torch.tensor(G, device = self.device), self(s_tau, a_tau))
        else:
            loss = self.loss_function(torch.tensor(G), self(s_tau, a_tau))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
