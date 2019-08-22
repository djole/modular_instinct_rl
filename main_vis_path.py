"""Visualise a trained model"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import episode_rollout, train_maml_like_for_trajectory


NUM_EPISODES = 40
NUM_UPDATES = 2
NUM_EXP = 5
MODEL_PATH = "./trained_models/pulled_from_server/20random_goals_instinct_module_danger_zone/individual_862.pt"


def vis_path(vis):
    """ Visualize the path """
    plt.figure()
    axis = plt.gca()
    for v in vis:
        path_rec = v[1]
        goal = v[3]

        pth = list(zip(*path_rec))
        axis.plot(*pth, color='green')
        axis.scatter(*goal, color='red')

    axis.add_patch(plt.Rectangle((0.2, 0.2), 0.1, 0.1, fc='r', alpha=0.1))
    axis.add_patch(plt.Rectangle((-0.3, 0.2), 0.1, 0.1, fc='r', alpha=0.1))
    axis.add_patch(plt.Rectangle((0.2, -0.3), 0.1, 0.1, fc='r', alpha=0.1))
    axis.add_patch(plt.Rectangle((-0.3, -0.3), 0.1, 0.1, fc='r', alpha=0.1))

    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)


    plt.show()
    print("")


def run(model, lr, unfreeze):
    """Run"""
    args = get_args()
    args.unfreeze_modules = unfreeze
    task_idx = 1
    # model_filename = "./trained_models/pulled_from_server/model995.pt"
    # model_filename = "./trained_models/pulled_from_server/maml_like_model_20episodes_lastGen436.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model597.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model977.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_985.pt"
    # m = Controller(2, 100, 2)
    # m = ControllerCombinator(2, 2, 100, 2)
    # env.seed(args.seed)

    # c_reward, reached, _, vis = episode_rollout(module, env, vis=True)
    c_reward, reached, vis = train_maml_like_for_trajectory(
        model, args, num_episodes=NUM_EPISODES, learning_rate=lr, num_updates=NUM_UPDATES, vis=True
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    vis_path(vis)
    return c_reward


def main():
    """Main"""
    import matplotlib.pyplot as plt

    num_exp = NUM_EXP

    model_filename = MODEL_PATH

    m1_orig = torch.load(model_filename)
    m1 = ControllerCombinator(2, 100, 2)
    m1.load_state_dict(m1_orig[0].state_dict())
    experiment1_fits = [run(m1, False, m1_orig[1]) for _ in range(num_exp)]
    experiment1_fits = list(zip(*experiment1_fits))
    with open("experiment1.list", "wb") as pckl_file1:
        pickle.dump(experiment1_fits, pckl_file1)


if __name__ == "__main__":
    main()
