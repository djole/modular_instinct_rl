from collections import deque
from copy import deepcopy

import torch
from gym.envs.registration import register

from arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.model import init_ppo, init_default_ppo
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.evaluation import evaluate
from navigation_2d import unpeele_navigation_env
from visualisations.vis_path import vis_heatmap, vis_path

#register(
#    id="Navigation2d-v0",
#    entry_point="navigation_2d:Navigation2DEnv",
#    max_episode_steps=200,
#    reward_threshold=0.0,
#)

ENV_NAME = "Navigation2d-v0"
NUM_PROC = 1


def train_maml_like_ppo(
    args, num_episodes=20, num_updates=1, run_idx=0, use_linear_lr_decay=False
):
    ###### Init boilerplate ######
    num_steps = num_episodes * 100
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    ###### Initialize the environment ######
    envs = make_vec_envs(
        ENV_NAME, args.seed, NUM_PROC, args.gamma, None, device, allow_early_resets=True
    )
    raw_env = unpeele_navigation_env(envs, 0)

    raw_env.set_arguments(args.rm_nogo, args.reduce_goals, True)
    new_task = raw_env.sample_tasks(run_idx)
    raw_env.reset_task(new_task[0])

    ###### Load the saved model and the learning rate ######
    actor_critic = init_ppo(envs)
    load_m = torch.load(
        "../trained_models/pulled_from_server/second_phase_instinct/2_deterministic_goals/large_zones_NOdistance2zones_PPO/individual_999.pt"
    )
    st_dct = load_m[0].state_dict()
    learning_rate = load_m[1]
    actor_critic.load_state_dict(st_dct)
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=learning_rate,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    ###### Initialize the storage datastructures ######
    rollouts = RolloutStorage(
        num_steps,
        NUM_PROC,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    fitnesses = []
    episode_path_record = []
    episodes_info = []

    ###### Start the main training loop ######
    for j in range(num_updates):
        episodes_info.clear()
        ###### Optionally, turn on the learning rate decay
        if use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        ###### Start the environment steps loop ######
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Observe the reward and next observation
            obs, reward, done, infos = envs.step(action)
            episode_path_record.append(raw_env._state)

            # Check if an episode finished and save the cumulative reward in the list
            for info in infos:
                if info["done"]:
                    episode_rewards.append(info["cummulative_reward"])
                    episodes_info.append((None, deepcopy(episode_path_record), None, info['goal']))
                    episode_path_record.clear()


            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            args.use_proper_time_limits,
        )

        ###### Plot the info part ######
        vis_path(episodes_info)
        # for s in range(1, 10):
        #    vis_path([vis_info], "eval_{}_{}".format(u_idx, s), s)
        # vis_path([vis_info])  # , "eval_{}".format(u_idx))
        # vis_heatmap(actor_critic)
        # vis_instinct_action(model)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #        and j % args.eval_interval == 0):
        ob_rms = utils.get_vec_normalize(envs).ob_rms
        fits, reached, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device)
        fitnesses.append(fits)
    return fitnesses[-1]


if __name__ == "__main__":
    args = get_args()
    args.reduce_goals = True
    envs = make_vec_envs(
        ENV_NAME, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )
    print("start the train function")
    train_maml_like_ppo(args,
        num_episodes=20, num_updates=6, run_idx=0, use_linear_lr_decay=False

    )
