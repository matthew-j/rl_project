import torch
import numpy as np

class Agent:
    def __init__(self, alpha, model, epsilon, num_actions, cuda):
        self.alpha = alpha
        self.model = model
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        self.epsilon = 1
        self.epsilon_decay = .99975
        self.epsilon_min = 0.1

        self.cuda = cuda
        if cuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        if self.cuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

    def get_action(self, state):
        self.model.eval()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        else:
            if self.cuda:
                action_values = self.model(state.to(self.device))
            else:
                action_values = self.model(state)
            action = torch.argmax(action_values, axis=0).item()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return action

    def __call__(self, state, action):
        if self.cuda:
            return self.model(state.to(self.device))[action]
        else:
            return self.model(state)[action]

    def update(self, G, s_tau, a_tau):
        self.model.train()
        if self.cuda:
            loss = G - self(s_tau.to(self.device), a_tau)
        else:
            loss = G - self(s_tau, a_tau)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
