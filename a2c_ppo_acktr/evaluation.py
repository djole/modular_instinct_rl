import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, env):
    eval_episode_rewards = []

    obs = env.reset()
    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size
    )
    eval_masks = torch.zeros(1, 1)

    done = False
    while not done:
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs.reshape(1,2)).float()
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
            )

        # Obser reward and next obs
        obs, _, done, *infos = env.step(action[0])

        eval_masks = torch.tensor([0.0] if done else [1.0], dtype=torch.float32)

        if done:
            eval_episode_rewards.append(infos[1])
            env.reset()

    return eval_episode_rewards[-1], infos[0], None
