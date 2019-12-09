""" 2D navigation environment """
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
import numpy as np

from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from math import pi, cos, sin, pow, sqrt

HORIZON = 100

START = [0.0, 0.0]

SMALL_NOGO_UPPER = 0.3
SMALL_NOGO_LOWER = 0.2
LARGE_NOGO_UPPER = 0.4
LARGE_NOGO_LOWER = 0.05


def dist_2_nogo(x, y):
    dist_fst = sqrt(pow(x - 0.25, 2) + pow(y - 0.25, 2))
    dist_snd = sqrt(pow(x + 0.25, 2) + pow(y - 0.25, 2))
    dist_trd = sqrt(pow(x - 0.25, 2) + pow(y + 0.25, 2))
    dist_frt = sqrt(pow(x + 0.25, 2) + pow(y + 0.25, 2))
    min_dist = min([dist_fst, dist_snd, dist_trd, dist_frt])
    return min_dist


def is_nogo(x, y, low, up):
    """Check if agent is in the nogo zone"""
    fst_square = (low < x < up) and (low < y < up)
    snd_square = (-up < x < -low) and (low < y < up)
    trd_square = (low < x < up) and (-up < y < -low)
    frt_square = (-up < x < -low) and (-up < y < -low)
    if fst_square or snd_square or trd_square or frt_square:
        return True
    return False

def intersection(p1, p2, q1, q2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    q1 = np.array(q1)
    q2 = np.array(q2)

    b = q1 - p1
    A = np.array([p2-p1, q1-q2]).transpose()
    try:
        segments = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return False
    return np.logical_and(segments >= 0, segments <= 1.0).all()

def is_stepping_over_square(p1, p2, sq_p1, sq_p3):
    sq_p2 = (sq_p1[0], sq_p3[1])
    sq_p4 = (sq_p1[1], sq_p3[0])

    int1 = intersection(p1, p2, sq_p1, sq_p2)
    int2 = intersection(p1, p2, sq_p2, sq_p3)
    int3 = intersection(p1, p2, sq_p3, sq_p4)
    int4 = intersection(p1, p2, sq_p4, sq_p1)
    return int1 or int2 or int3 or int4

def is_crossing_nogo(prev_point, point, low, high):
    current_is_nogo = is_nogo(point[0], point[1], low, high)
    if current_is_nogo:
        return True
    if is_nogo(prev_point[0], prev_point[1], low, high) and not current_is_nogo:
        return False
    else:
        # Check squares
        fst = is_stepping_over_square(prev_point, point, (low, low), (high, high))
        snd = is_stepping_over_square(prev_point, point, (-low, low), (-high, high))
        trd = is_stepping_over_square(prev_point, point, (low, -low), (high, -high))
        fourth = is_stepping_over_square(prev_point, point, (-low, -low), (-high, -high))
        return fst or snd or trd or fourth


def unpeele_navigation_env(env, envIdx):
    if isinstance(env, Navigation2DEnv):
        return env
    elif isinstance(env, DummyVecEnv) or isinstance(env, ShmemVecEnv):
        return unpeele_navigation_env(env.envs[envIdx], envIdx)
    else:
        try:
            env = env.env
        except:
            env = env.venv
        return unpeele_navigation_env(env, envIdx)

class Navigation2DEnv(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py

    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal 
    position (ie. the reward is `-distance`). The 2D navigation tasks are 
    generated by sampling goal positions from the uniform distribution 
    on [-0.5, 0.5]^2.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, task={}, rm_nogo=False, reduced_sampling=False, rm_dist_to_nogo=False, nogo_large=False):
        super(Navigation2DEnv, self).__init__()

        obs_shape = (2,) if rm_dist_to_nogo else (3,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get("goal", np.zeros(2, dtype=np.float32))
        self._state = np.array(START)  # np.zeros(2, dtype=np.float32)
        self._previous_state = np.array(START)
        self.seed()
        self.horizon = HORIZON
        self.cummulative_reward = 0
        self.episode_x_path = []
        self.episode_y_path = []

        # An option to remove no-go zones for baseline purposes
        self.rm_nogo = rm_nogo
        # An option that cycles only through two goals
        self.reduced_sampling = reduced_sampling
        # An option that removes the information from state about the distance to the center of a nogo zone
        self.rm_dist_to_nogo = rm_dist_to_nogo

        self.task_sequence = [[0.35, 0.45],[-0.45, -0.23]]
        # Values that define the boundaries of no-go zones
        self.nogo_lower = LARGE_NOGO_LOWER if nogo_large else SMALL_NOGO_LOWER
        self.nogo_upper = LARGE_NOGO_UPPER if nogo_large else SMALL_NOGO_UPPER
    # for info in infos:
    #    if 'episode' in info.keys() and info['done']:
    #        episode_rewards.append(info['episode']['r'])

    def _sample_ring_task(self):
        radius = self.np_random.uniform(0.3, 0.5, size=(1, 1))[0][0]
        alpha = self.np_random.uniform(0.0, 1.0, size=(1, 1)) * 2 * pi
        alpha = alpha[0][0]
        goal = np.array([[radius * cos(alpha), radius * sin(alpha)]])
        return goal

    def _sample_square_wth_nogo_zone(self):
        rand_x = self.np_random.uniform(-0.5, 0.5, size=(1, 1))[0][0]
        if rand_x <= self.nogo_upper and rand_x >= -self.nogo_upper:
            # If random x could be in the no-go zone
            # Sample randomly from four slices
            dart = self.np_random.uniform(0.0, 1.0, size=(1, 1))[0][0]
            if dart <= 0.5:
                rand_y = self.np_random.uniform(-0.5, -self.nogo_upper, size=(1, 1))[0][0]
            elif dart > 0.5:
                rand_y = self.np_random.uniform(self.nogo_upper, 0.5, size=(1, 1))[0][0]
        else:
            rand_y = self.np_random.uniform(-0.5, 0.5, size=(1, 1))[0][0]

        goal = np.array([[rand_x, rand_y]])
        return goal

    def _sample_predetermined(self, idx):
        idx = idx % len(self.task_sequence)
        goals = [self.task_sequence[idx]]
        return goals

    def seed(self, seed=None):
        seed = None
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, idx):
        goals = self._sample_predetermined(idx) if self.reduced_sampling else self._sample_square_wth_nogo_zone()
        # goals = self.np_random.uniform(-0.5, 0.5, size=(1, 2))
        # goals = np.array(self.task_sequence)
        tasks = [{"goal": goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task["goal"]

    def reset(self):
        self._state = np.array(START)  # np.zeros(2, dtype=np.float32)
        self.horizon = HORIZON
        self.cummulative_reward = 0
        self.episode_x_path.clear()
        self.episode_y_path.clear()
        self.episode_x_path.append(self._state[0])
        self.episode_y_path.append(self._state[1])

        d2ng = dist_2_nogo(self._state[0], self._state[1])

        if self.rm_dist_to_nogo:
            state_info = self._state
        else:
            state_info = np.append(self._state, [d2ng])

        return state_info

    def step(self, action):

        np.copyto(self._previous_state, self._state)
        action = np.clip(action, -0.1, 0.1)
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()

        assert self.action_space.contains(action)
        self._state = self._state + action

        delta_x = self._state[0] - self._goal[0]
        delta_y = self._state[1] - self._goal[1]
        reward = -np.sqrt(delta_x ** 2 + delta_y ** 2)

        # Check if the x and y are in the no-go zone
        # If yes, punish the agent.
        if not self.rm_nogo and is_crossing_nogo(self._previous_state, self._state, self.nogo_lower, self.nogo_upper):
            reward -= 10

        d2ng = dist_2_nogo(self._state[0], self._state[1])

        reached = (np.abs(delta_x) < 0.01) and (np.abs(delta_y) < 0.01)
        done = reached or self.horizon <= 0
        self.horizon -= 1
        self.cummulative_reward += reward

        self.episode_x_path.append(self._state[0])
        self.episode_y_path.append(self._state[1])

        if self.rm_dist_to_nogo:
            state_info = self._state
        else:
            state_info = np.append(self._state, [d2ng])

        info_dict = {'reached' : reached,
            'cummulative_reward':self.cummulative_reward,
            'goal':self._goal,
            'done':done
                     }
        if done:
            info_dict['path'] = list(zip(self.episode_x_path, self.episode_y_path))

        return (
            state_info,
            reward,
            done,
            info_dict,
        )

    def render_episode(self):
        plt.figure()
        plt.plot(self.episode_x_path, self.episode_y_path)
        plt.plot(self._goal[0], self._goal[0], "r*")
        plt.show()
