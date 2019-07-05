"""Visualise a trained model"""
import pickle
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import episode_rollout, train_maml_like


NUM_EPISODES = 40
NUM_UPDATES = 4
NUM_EXP = 100


def vis_path(path_rec, action_vec, model_info, goal):
    """ Visualize the path """
    plt.figure()

    unzipped_info = list(zip(*model_info))
    unzipped_info[0]
    votes = unzipped_info[1]
    votes = list(map(lambda x: np.split(x, 2), votes))
    votes1 = list(map(lambda x: x[0], votes))
    votes2 = list(map(lambda x: x[1], votes))

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


def run(model, learning_rate, unfreeze):
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

    import numpy as np

    # c_reward, reached, _, vis = episode_rollout(module, env, vis=True)
    c_reward, reached, vis = train_maml_like(
        model,
        args,
        learning_rate,
        num_episodes=NUM_EPISODES,
        num_updates=NUM_UPDATES,
        vis=True,
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    # vis_path(vis[1][:-1], vis[0], vis[2], vis[3])
    return c_reward


def run_for_pool(_, m):
    return run(m, False)


def calc_fitness(model_filename_base, savefile, tuple_pckl=False):
    num_exp = NUM_EXP
    m_base_orig = torch.load(model_filename_base)
    if tuple_pckl:
        m_base_orig = m_base_orig[0]
    m_base = ControllerCombinator(2, 4, 100, 2, 2, sees_inputs=False)
    m_base.load_state_dict(m_base_orig.state_dict())

    rfp = partial(run_for_pool, m=m_base)
    with Pool(20) as pool:
        experiment_base_fits = list(pool.map(rfp, range(num_exp)))

    experiment_base_fits = list(zip(*experiment_base_fits))
    with open(savefile, "wb") as pckl_file1:
        pickle.dump(experiment_base_fits, pckl_file1)

    return experiment_base_fits


def main():
    """Main"""
    import matplotlib.pyplot as plt

    num_exp = NUM_EXP

    model_filename_base = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_880.pt"
    model_filename_ring003 = "./trained_models/pulled_from_server/20random_RING_goals_20episode_monolith_multiplexor/ring_sample_003.pt"
    model_filename_ring001 = "./trained_models/pulled_from_server/20random_RING_goals_20episode_monolith_multiplexor/ring_sample_001.pt"
    model_filename_ring009 = "./trained_models/pulled_from_server/20random_RING_goals_20episode_monolith_multiplexor/ring_sample_009.pt"
    model_filename_ring_elr = "./trained_models/pulled_from_server/20random_RING_goals_20episode_monolith_multiplexor/ring_sample_evLR.pt"

    experiment_base_fits = calc_fitness(
        model_filename_base, "experiment_fits_BASE.pckl"
    )

    experiment_ring003_fits = calc_fitness(
        model_filename_ring003, "experiment_fits_ring003.pckl"
    )

    experiment_ring001_fits = calc_fitness(
        model_filename_ring001, "experiment_fits_ring001.pckl"
    )

    experiment_ring009_fits = calc_fitness(
        model_filename_ring009, "experiment_fits_ring009.pckl"
    )

    experiment_ring_elr_fits = calc_fitness(
        model_filename_ring_elr, "experiment_fits_ringELR.pckl", True
    )

    assert len(experiment_base_fits) == len(experiment_ring001_fits)
    fig1, axs = plt.subplots(ncols=len(experiment_base_fits), figsize=(30, 15))

    try:
        labels = ["base train", "ring003", "ring001", "ring009", "ring003\nELR"]
        for i, ax in enumerate(axs):
            ax.set_title("Eval. fitness {} updates".format(i))
            ax.set_ylim(-100, 0)
            ax.boxplot(
                [
                    experiment_base_fits[i],
                    experiment_ring003_fits[i],
                    experiment_ring001_fits[i],
                    experiment_ring009_fits[i],
                    experiment_ring_elr_fits[i],
                ],
                labels=labels,
                showmeans=True,
            )
    except TypeError:
        axs.set_title("Eval. fitness {} updates".format(0))
        axs.boxplot(
            [
                experiment_base_fits[0],
                experiment_ring003_fits[0],
                experiment_ring001_fits[0],
                experiment_ring009_fits[0],
                experiment_ring_elr_fits[0],
            ],
            labels=labels,
            showmeans=True,
        )

    plt.savefig("./fig.jpg")
    # plt.show()


if __name__ == "__main__":
    main()
