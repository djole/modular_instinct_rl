import argparse
import navigation_2d
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from model import ControllerCombinator

import copy

from model import Controller, ControllerCombinator

EPS = np.finfo(np.float32).eps.item()


def select_model_action(model, state):
    state = torch.from_numpy(state).float()
    action, action_log_prob, debug_info = model(state)
    #return action.item()
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
    
    #new_task = env.sample_tasks()
    #env.reset_task(new_task[goal_index])

    state = env.reset()
    cummulative_reward = 0
    rewards = []
    action_log_probs = []
    
    ######
    # Visualisation elements
    action_records = list()
    path_records = list()
    if vis: path_records.append(env._state)
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

    return cummulative_reward, reached, (rewards, action_log_probs), (action_records, path_records, debug_info_records, env._goal)


def train_maml_like(init_model, env, args, num_episodes=20, num_updates=1, vis=False):
    env = navigation_2d.Navigation2DEnv()
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator):
        optimizer = torch.optim.Adam(model.get_combinator_params(args.unfreeze_modules), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rewards = []
    action_log_probs = []

    fitness_list = []
    ### evaluate for the zero updates
    if vis:
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    for u_idx in range(num_updates):
        ### Train
        model.deterministic = False
        for _ in range(num_episodes):
            _, reached, (rewards_, action_log_probs_), _ = episode_rollout(model, env, False)
            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)

        # Reduce the learning rate of the optimizer by half in the first iteration
        if u_idx == 0 and vis:
            new_learning_rate = args.lr / 2.
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    ret_fit = fitness_list if vis else fitness_list[-1]
    return ret_fit, reached, vis_info
