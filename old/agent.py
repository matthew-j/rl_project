import torch
from torch.distributions import Categorical
import numpy as np

class Agent:
    def __init__(self, alpha, model, epsilon, num_actions, batch_size, device):
        self.device = device
        self.alpha = alpha
        self.model = model
        self.model.to(self.device)
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_function = torch.nn.SmoothL1Loss()

        self.epsilon = 1.000000
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.batch_size = batch_size
    
    def get_epsilon(self):
        return self.epsilon

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.to(self.device)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        self.model.eval()

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        else:
            action_values = self.model(state.to(self.device))
            return torch.argmax(action_values, axis=1).item()

    def __call__(self, state, action):
        batches = state.size()[0]
        return self.model(state.to(self.device))[
            np.arange(0, batches), action
            ]

    def update(self, G, s_tau, a_tau):
        self.model.train()
        loss = self.loss_function(torch.tensor(G, device = self.device), self(s_tau, a_tau))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
