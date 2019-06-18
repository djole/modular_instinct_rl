"""Visualise a trained model"""
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import episode_rollout, train_maml_like


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
    env = navigation_2d.Navigation2DEnv()
    # env.seed(args.seed)

    ## DEBUGGING
    # for p in m.parameters():
    #    p.data = torch.randn_like(p.data) * 10
    ###

    import numpy as np

    # c_reward, reached, _, vis = episode_rollout(module, env, vis=True)
    c_reward, reached, vis = train_maml_like(
        model, env, args, num_episodes=40, num_updates=4, vis=True
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    # vis_path(vis[1][:-1], vis[0], vis[2], vis[3])
    return c_reward


def main():
    """Main"""
    import matplotlib.pyplot as plt

    num_exp = 100
    model_filename2 = "./trained_models/pulled_from_server/20random_goals_monolith_network/individual_829.pt"
    m2_orig = torch.load(model_filename2)
    m2 = ControllerMonolithic(2, 100, 2)
    m2.load_state_dict(m2_orig.state_dict())
    experiment2_fits = [run(m2, True) for _ in range(num_exp)]
    experiment2_fits = list(zip(*experiment2_fits))

    model_filename = "./trained_models/pulled_from_server/20random_goals8modules20episode_lr01/individual_65.pt"
    m1 = torch.load(model_filename)
    experiment1_fits = [run(m1, False) for _ in range(num_exp)]
    experiment1_fits = list(zip(*experiment1_fits))

    model_filename3 = "./trained_models/pulled_from_server/20random_goals4modules20episode_lr01/individual_707.pt"
    m3 = torch.load(model_filename3)
    experiment3_fits = [run(m3, False) for _ in range(num_exp)]
    experiment3_fits = list(zip(*experiment3_fits))

    assert len(experiment1_fits) == len(experiment2_fits) and len(
        experiment2_fits
    ) == len(experiment3_fits)
    fig1, axs = plt.subplots(ncols=len(experiment1_fits))

    try:
        for i, ax in enumerate(axs):
            ax.set_title("Evaluation fitness after {} updates, lr=0.1".format(i))
            ax.boxplot(
                [experiment1_fits[i], experiment2_fits[i], experiment3_fits[i]],
                labels=[
                    "8 modules\n2 outputs per mod",
                    "monolith",
                    "4 modules\n2 outputs per mod",
                ],
                showmeans=True,
            )
    except TypeError:
        axs.set_title("Evaluation fitness after {} updates".format(0))
        axs.boxplot(
            [experiment1_fits[0], experiment2_fits[0], experiment3_fits[0]],
            labels=[
                "8 modules\n2 outputs per mod",
                "monolith",
                "4 modules\n2 outputs per mod",
            ],
            showmeans=True,
        )

    plt.savefig("./fig.jpg")
    plt.show()


if __name__ == "__main__":
    main()
