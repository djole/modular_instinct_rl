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

import copy

from model import Controller, ControllerCombinator

EPS = np.finfo(np.float32).eps.item()


def select_model_action(model, state):
    state = torch.from_numpy(state).float()
    action, action_log_prob, debug_info = model(state)
    #return action.item()
    return action.detach().numpy(), action_log_prob, debug_info

def update_policy(model, optimizer, args, rewards, log_probs):
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

def episode_rollout(init_model, args, env, rollout_index, num_updates=1, adapt=True, num_episodes=20, vis=False):
    
    new_task = env.sample_tasks()
    env.reset_task(new_task[rollout_index])

    model = copy.deepcopy(init_model)

    optimizer = torch.optim.Adam(model.get_combinator_params(), lr=args.lr)

    state = env.reset()
    cummulative_reward = 0
    rewards = []
    action_log_probs = []
    
    ######
    # Visualisation elements
    action_records = list()
    path_records = list()
    debug_info_records = list()
    # ---------------------
    reached_global = 0

    for up_idx in range(num_updates):
        for ep_idx in range(num_episodes):
            while True:
                action, action_log_prob, debug_info = select_model_action(model, state)
                action = action.flatten()
                state, reward, done, reached, _, _ = env.step(action)
                cummulative_reward += reward

                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                reached_global += 1 if reached else 0
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

        if adapt:
            assert len(rewards) > 1 and len(action_log_probs) > 1
            update_policy(model, optimizer, args, rewards, action_log_probs)
            rewards.clear()
            action_log_probs.clear()

    return cummulative_reward, reached_global, (action_records, path_records, debug_info_records)

def main(args):
    fig = plt.figure() 
    plt.ion()
    plt.show()
    policy = Controller(2, 100, 2)
    running_reward = 0
    reward_buffer = deque(maxlen=10)
    rewards = []
    action_log_probs = []

    env = navigation_2d.Navigation2DEnv()
    env.seed(args.seed)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)

    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 200):  # Don't infinite loop while learning
            action, action_log_prob, _ = select_model_action(policy, state)
            action = action.flatten()
            state, reward, done, reached, _, _ = env.step(action)
            if args.render:
                env.render()
            rewards.append(reward)
            action_log_probs.append(action_log_prob)
            ep_reward += reward
            if done:
                if reached:
                    print("Reached! cummulative reward {}".format(env.cummulative_reward))
                    #input()
                reward_buffer.append(env.cummulative_reward)
                env.reset()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        update_policy(policy, optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward form last ten episodes: {:.2f}'.format(
                  i_episode, ep_reward, sum(reward_buffer)/float(len(reward_buffer))))
            #env.render()

        if running_reward > 0:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    from arguments import get_args
    main(get_args())
