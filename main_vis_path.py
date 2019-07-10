"""Visualise a trained model"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import episode_rollout, train_maml_like


NUM_EPISODES = 40
NUM_UPDATES = 2
NUM_EXP = 5
MODEL_PATH = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_880.pt"


def vis_path(path_rec, action_vec, model_info, goal):
    """ Visualize the path """
    plt.figure()

    unzipped_info = list(zip(*model_info))
    # unzipped_info[0]
    votes = unzipped_info[1]
    votes = list(map(lambda x: np.split(x, 2), votes))
    # votes1 = list(map(lambda x: x[0], votes))
    # votes2 = list(map(lambda x: x[1], votes))

    prec = list(zip(*path_rec))
    # --------------------
    # plot combinator vectors
    # --------------------
    # combinator_mean_xs, combinator_mean_ys = zip(*combinator_means)
    # axis = plt.gca()
    # axis.quiver(prec[0], prec[1], combinator_mean_xs, combinator_mean_ys,
    #            angles='xy', scale_units='xy', scale=1, color='green', headaxislength=0, headlength=0)

    # --------------------
    # plot module 1 vectors
    # --------------------
    # votes1_xs, votes1_ys = zip(*votes1)
    # axis = plt.gca()
    # axis.quiver(prec[0], prec[1], votes1_xs, votes1_ys,
    #            angles='xy', scale_units='xy', scale=1, color='blue', headaxislength=0, headlength=0)

    #### --------------------
    #### plot module 2 vectors
    #### --------------------
    # votes2_xs, votes2_ys = zip(*votes2)
    # axis = plt.gca()
    # axis.quiver(prec[0], prec[1], votes2_xs, votes2_ys,
    #            angles='xy', scale_units='xy', scale=1, color='red', headaxislength=0, headlength=0)

    # --------------------
    # plot path
    # --------------------

    plt.plot(*prec, "-o")
    plt.plot(goal[0], goal[1], "r*")
    plt.show()


def run(model, unfreeze):
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
    c_reward, reached, vis = train_maml_like(
        model, args, num_episodes=NUM_EPISODES, num_updates=NUM_UPDATES, vis=True
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    vis_path(vis[1][:-1], vis[0], vis[2], vis[3])
    return c_reward


def main():
    """Main"""
    import matplotlib.pyplot as plt

    num_exp = NUM_EXP

    model_filename = MODEL_PATH

    m1_orig = torch.load(model_filename)
    m1 = ControllerCombinator(2, 4, 100, 2, 2, sees_inputs=False)
    m1.load_state_dict(m1_orig.state_dict())
    experiment1_fits = [run(m1, False) for _ in range(num_exp)]
    experiment1_fits = list(zip(*experiment1_fits))
    with open("experiment1.list", "wb") as pckl_file1:
        pickle.dump(experiment1_fits, pckl_file1)


if __name__ == "__main__":
    main()
