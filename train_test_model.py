""" Module for training functions """
import copy
from collections import deque

import numpy as np
import torch

import navigation_2d
from model import ControllerCombinator, ControllerNonParametricCombinator
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.evaluation import evaluate

EPS = np.finfo(np.float32).eps.item()


def select_model_action(model, state):
    state_ = state
    state_ = torch.from_numpy(state_).float()
    # dist_2_nogo = torch.tensor([dist_2_nogo])
    # model_input = torch.cat([position, dist_2_nogo])
    action, action_log_prob, debug_info = model(state_)
    # return action.item()
    return action.detach().numpy(), action_log_prob, debug_info


def update_policy(optimizer, args, rewards, log_probs):
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


def episode_rollout(model, env, vis=False):

    # new_task = env.sample_tasks()
    # env.reset_task(new_task[goal_index])

    state = env.reset()
    cummulative_reward = 0
    rewards = []
    action_log_probs = []

    ######
    # Visualisation elements
    action_records = list()
    path_records = list()
    if vis:
        path_records.append(env._state)
    debug_info_records = list()
    # ---------------------

    while True:
        action, action_log_prob, debug_info = select_model_action(model, state)
        action = action.flatten()
        state, reward, done, reached, _, _ = env.step(action)
        cummulative_reward += reward

        rewards.append(reward)
        action_log_probs.append(action_log_prob)
        ######
        # Visualisation elements
        if vis:
            action_records.append(action)
            path_records.append(env._state)
            debug_info_records.append(debug_info)
        # ---------------------
        if done:
            env.reset()
            break

    return (
        cummulative_reward,
        reached,
        (rewards, action_log_probs),
        (action_records, path_records, debug_info_records, env._goal),
    )


def train_maml_like_ppo(
    init_model,
    args,
    learning_rate,
    num_episodes=20,
    num_updates=1,
    vis=False,
    run_idx=0,
):

    torch.set_num_threads(1)

    env = navigation_2d.Navigation2DEnv(
        rm_nogo=args.rm_nogo, reduced_sampling=args.reduce_goals, sample_idx=run_idx
    )
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    # actor_critic = Policy(
    #    env.observation_space.shape,
    #    env.action_space,
    #    base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic = copy.deepcopy(init_model)

    agent = PPO(
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

    num_steps = env.horizon

    rollouts = RolloutStorage(
        num_steps,
        1,
        env.observation_space.shape,
        env.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs).float()
    rollouts.obs[0].copy_(obs)

    fits = []

    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, *infos = env.step(action[0])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done else [1.0]])
            bad_masks = torch.FloatTensor(
                [[1.0]]
            )

            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs)
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                torch.Tensor([reward]),
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
            use_gae=False,
            gamma=args.gamma,
            gae_lambda=0.95,
            use_proper_time_limits=False,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        fits.append(evaluate(actor_critic, env))

    return fits[-1]


def train_maml_like(
    init_model,
    args,
    learning_rate,
    num_episodes=20,
    num_updates=1,
    vis=False,
    run_idx=0,
):
    env = navigation_2d.Navigation2DEnv(
        rm_nogo=args.rm_nogo, reduced_sampling=args.reduce_goals, sample_idx=run_idx
    )
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator) or isinstance(
        model, ControllerNonParametricCombinator
    ):
        optimizer = torch.optim.Adam(model.get_combinator_params(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rewards = []
    action_log_probs = []

    fitness_list = []
    ### evaluate for the zero updates
    if vis:
        model.controller.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    for u_idx in range(num_updates):
        avg_exploration_fitness = 0
        ### Train
        model.controller.deterministic = False
        for ne in range(num_episodes):
            exploration_fitness, reached, (
                rewards_,
                action_log_probs_,
            ), _ = episode_rollout(model, env, False)
            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)
            avg_exploration_fitness = (
                exploration_fitness + ne * avg_exploration_fitness
            ) / (ne + 1)

        # Reduce the learning rate of the optimizer by half in the first iteration
        if u_idx > 0:
            new_learning_rate = learning_rate / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.controller.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    rm_exp_fit = args.rm_nogo or args.rm_exploration_fit
    avg_exploration_fitness = 0.0 if rm_exp_fit else avg_exploration_fitness
    avg_exploitation_fitness = sum(fitness_list) / num_updates
    ret_fit = (
        fitness_list if vis else avg_exploitation_fitness + avg_exploration_fitness
    )
    return ret_fit, reached, vis_info


def train_maml_like_for_trajectory(
    init_model, args, learning_rate, num_episodes=20, num_updates=1, vis=False
):
    # TODO Remove this function, this is bad programming
    assert False, "Obsolete piece of code, remove it!"
    env = navigation_2d.Navigation2DEnv(args.rm_nogo, args.reduce_goals)
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator):
        optimizer = torch.optim.Adam(model.get_combinator_params(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rewards = []
    action_log_probs = []

    fitness_list = []
    ### evaluate for the zero updates
    vis_info_collection = []
    if vis:
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        vis_info_collection.append(vis_info)
        fitness_list.append(fitness)

    for u_idx in range(num_updates):
        ### Train
        model.deterministic = False
        for _ in range(num_episodes):
            _, reached, (rewards_, action_log_probs_), vis_info = episode_rollout(
                model, env, True
            )
            vis_info_collection.append(vis_info)

            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)

        # Reduce the learning rate of the optimizer by half in the first iteration
        if u_idx == 0 and vis:
            new_learning_rate = args.lr / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        vis_info_collection.append(vis_info)
        fitness_list.append(fitness)

    ret_fit = fitness_list if vis else fitness_list[-1]
    return ret_fit, reached, vis_info_collection
