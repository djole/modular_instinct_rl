import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

HORIZON = 100
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
    def __init__(self, task={}):
        super(Navigation2DEnv, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()
        self.horizon = HORIZON
        self.cummulative_reward = 0
        self.episode_x_path = []
        self.episode_y_path = []

        self.task_sequence = [[0.7, 0.35], [-0.7, -0.35]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self):
        #goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        goals = np.array(self.task_sequence)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        self.horizon = HORIZON
        self.cummulative_reward = 0
        self.episode_x_path.clear()
        self.episode_y_path.clear()
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)

        reached = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))
        done = reached or self.horizon <= 0
        self.horizon -= 1
        self.cummulative_reward += reward

        self.episode_x_path.append(self._state[0])
        self.episode_y_path.append(self._state[1])

        return self._state, reward, done, reached, self.cummulative_reward, self._task
    
    def render_episode(self):
        fig = plt.figure()
        plt.plot(self.episode_x_path, self.episode_y_path)
        plt.plot(self._goal[0], self._goal[1], 'r*')
        plt.show()
