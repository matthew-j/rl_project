import time
import torch
from models import ActorCriticNN
from environment import generate_env

def evaluate(target_model, num_episodes, render):
    env = generate_env()
    model = ActorCriticNN(env.observation_space.shape, env.action_space.n)
    model.eval()


    for i in range(num_episodes):
        cur_state = env.reset()
        model.load_state_dict(target_model.state_dict())

        done = True
        episode_reward = 0

        while True:
            if done:
                cell_state = torch.zeros(1, 256)
                hidden_state = torch.zeros(1, 256)
            else:
                cell_state = cell_state.detach()
                hidden_state = hidden_state.detach()

            cur_state = torch.tensor([cur_state.__array__().tolist()])
            with torch.no_grad():
                action, (hidden_state, cell_state) = model.act((cur_state, (hidden_state, cell_state)))
            cur_state, reward, done, info = env.step(action)
            if render:
                env.render()
            episode_reward += reward

            if done:
                break

        # Wait a bit until model may be different
        print(f"Episode {i} reward: {episode_reward}")
        time.sleep(60)

            


