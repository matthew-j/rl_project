from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import cv2
from torch import nn
import torch
import time

class MarioNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(0),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    def forward(self, input):
        return self.online(input)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
model = MarioNet((3, 84, 84), 8)
done = True
update_freq = 12
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

# Load
device = torch.device("cuda")
model.to(device)

episodes = 10
start = time.time()
for episode in range(episodes):
    print("Episode: ", episode)
    env.reset()

    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        state_resize = cv2.resize(state, (84, 84), cv2.INTER_AREA)
        crop_img = img
        exit()
        if step % update_freq == 0:
            res = model(torch.tensor(state_resize).reshape(3, 84, 84).float().to(device))
            res = model(torch.tensor(state_resize).reshape(3, 84, 84).float().to(device))
            loss = 4 * res[0] + 100 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env.close()
end = time.time()
print("Time taken:", end - start)
print("Episodes: ", episodes)