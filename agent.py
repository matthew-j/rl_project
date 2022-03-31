import torch
import numpy as np

class Agent:
    def __init__(self, alpha, model, epsilon, num_actions):
        self.alpha = alpha
        self.model = model
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def get_action(self, state):
        self.model.eval()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        action_values = self.model(state)
        action = torch.argmax(action_values, axis=1).item()

        return action

    def __call__(self, state, action):
        return self.model(state)[action]

    def update(self, G, s_tau, a_tau):
        self.model.train()
        loss = G - self(s_tau, a_tau)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
