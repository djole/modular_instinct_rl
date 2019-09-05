""" Module for training functions """
import copy

import numpy as np
import torch

import navigation_2d
from model import ControllerCombinator, ControllerNonParametricCombinator

EPS = np.finfo(np.float32).eps.item()


def select_model_action(model, state):
    position, dist_2_nogo = state
    position = torch.from_numpy(position).float()
    dist_2_nogo = torch.tensor([dist_2_nogo])
    model_input = torch.cat([position, dist_2_nogo])
    action, action_log_prob, debug_info = model(model_input)
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


def train_maml_like(
    init_model, args, learning_rate, num_episodes=20, num_updates=1, vis=False
):
    env = navigation_2d.Navigation2DEnv(rm_nogo=args.rm_nogo)
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator) or isinstance(model, ControllerNonParametricCombinator):
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

    avg_exploration_fitness = 0
    for u_idx in range(num_updates):
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
    avg_exploitation_fitness = 0.0 if rm_exp_fit else sum(fitness_list) / num_updates
    ret_fit = (
        fitness_list if vis else avg_exploitation_fitness + avg_exploration_fitness
    )
    return ret_fit, reached, vis_info


def train_maml_like_for_trajectory(
    init_model, args, learning_rate, num_episodes=20, num_updates=1, vis=False
):
    # TODO Remove this function, this is bad programming
    assert False, "Obsolete piece of code, remove it!"
    env = navigation_2d.Navigation2DEnv(args.rm_nogo)
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
