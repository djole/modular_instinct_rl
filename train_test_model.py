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

from model import Controller, ControllerCombinator

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = navigation_2d.Navigation2DEnv()
env.seed(args.seed)
GOAL = {'goal' : [0.75, 0.3]}
env.reset_task(GOAL)
torch.manual_seed(args.seed)

EPS = np.finfo(np.float32).eps.item()


def select_model_action(model, state):
    state = torch.from_numpy(state).float()
    action, action_log_prob = model(state)
    #return action.item()
    return action.detach().numpy(), action_log_prob

def update_policy(model, rewards, log_probs, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.get_combinator_params(), lr=learning_rate)
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

def episode_rollout(model, rollout_index, max_steps=100, sample_goal=False, adapt=True, update_gap=10):
    if sample_goal:
        new_task = env.sample_tasks(1)
        env.reset_task(new_task[0])

    state = env.reset()
    cummulative_reward = 0
    rewards = []
    action_log_probs = []
    for st in range(max_steps):
        for ii in range(update_gap):
            action, action_log_prob = select_model_action(model, state)
            action = action.flatten()
            state, reward, done, reached, _, _ = env.step(action)
            cummulative_reward += reward

            rewards.append(reward)
            action_log_probs.append(action_log_prob)
            
            if done:
                break
        if done:
            break

        if adapt:
            assert len(rewards) > 1 and len(action_log_probs) > 1
            update_policy(model, rewards, action_log_probs)
            rewards.clear()
            action_log_probs.clear()

    return cummulative_reward, reached

def main():
    fig = plt.figure() 
    plt.ion()
    plt.show()
    policy = Controller(2, 100, 2)
    running_reward = 0
    reward_buffer = deque(maxlen=10)
    rewards = []
    action_log_probs = []
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 200):  # Don't infinite loop while learning
            action, action_log_prob = select_model_action(policy, state)
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
        update_policy(policy, rewards, action_log_probs, learning_rate=0.0001)
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
    main()
