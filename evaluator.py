import time
import torch
import logging

from environment import generate_env

def evaluate(algorithm_name, target_model, eval_model, T, Tmax, render):
    logging.basicConfig(filename=f"logs/{algorithm_name}.log", level=logging.DEBUG)
    env = generate_env()
    eval_model.eval()
    savecnt = 0

    while(T.data < Tmax):
        cur_state = env.reset()
        eval_model.load_state_dict(target_model.state_dict())

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
                action, _, (hidden_state, cell_state) = eval_model.act((cur_state, (hidden_state, cell_state)))
            cur_state, reward, done, info = env.step(action)
            if render:
                env.render()
            episode_reward += reward

            if done:
                break

        # Wait a bit until model may be different
        step_num = T.data
        info_string = f"Time: {time.asctime()} Step: {step_num} Reward: {episode_reward}"
        logging.info(info_string)
        print(info_string)
        
        savecnt += 1
        if savecnt % 8 == 0:
            torch.save(eval_model.state_dict(), f"saves/{algorithm_name}{step_num}.pt")
        time.sleep(30)
