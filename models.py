import torch
from torch import nn, argmax
import torch.nn.functional as F

class MiniCnn(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    def forward(self, input):
        return self.model(input)

class ActorCriticNN(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(3136, 512)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(512, 256)

        self.q_linear = nn.Linear(256, output_dim)
        self.value_linear = nn.Linear(256, 1)
        self.apply(self.init_weights)

    def forward(self, inputs):
        x0, (hidden_state, cell_state) = inputs
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = self.flatten(x3)
        x5 = self.linear1(x4)
        hidden_state, cell_state = self.lstm(x5, (hidden_state, cell_state))

        q_values = F.softmax(self.q_linear(hidden_state), dim=-1)
        state_values = self.value_linear(hidden_state)

        return q_values, state_values, (hidden_state, cell_state)

    def act(self, inputs):
        q_values, _, (hidden_state, cell_state) = self(inputs)
        return argmax(q_values).item(), (hidden_state, cell_state)

class ActorCriticNN4Layers(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.linear1 = nn.Linear(1152, 512)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTMCell(512, 256)

        self.q_linear = nn.Linear(256, output_dim)
        self.value_linear = nn.Linear(256, 1)
        self.apply(self.init_weights)

    def forward(self, inputs):
        x0, (hidden_state, cell_state) = inputs
        x1 = F.elu(self.conv1(x0))
        x2 = F.elu(self.conv2(x1))
        x3 = F.elu(self.conv3(x2))
        x3 = F.elu(self.conv4(x3))
        x4 = self.flatten(x3)
        x5 = self.linear1(x4)
        hidden_state, cell_state = self.lstm(x5, (hidden_state, cell_state))

        q_values = F.softmax(self.q_linear(hidden_state), dim=-1)
        state_values = self.value_linear(hidden_state)

        return q_values, state_values, (hidden_state, cell_state)

    def act(self, inputs):
        q_values, _, (hidden_state, cell_state) = self(inputs)
        return argmax(q_values).item(), (hidden_state, cell_state)