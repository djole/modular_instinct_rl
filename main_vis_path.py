"""Visualise a trained model"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import train_maml_like, select_model_action
from navigation_2d import dist_2_nogo

NUM_EPISODES = 40
NUM_UPDATES = 1
NUM_EXP = 5
MODEL_PATH = "./trained_models/pulled_from_server/20random_goals_instinct_module_danger_zone/larger_pop_smaller_punishment/individual_175.pt"


def vis_instinct_action(model):
    input_xs = get_mesh()
    z = [select_model_action(model, (x, dist_2_nogo(*x)))[2] for x in input_xs]
    plt.figure()
    axis = plt.gca()
    # x, y = input_xs.transpose()[0], input_xs.transpose()[1]
    # x = np.reshape(x, (40, 40))
    # y = np.reshape(y, (40, 40))
    # z = np.reshape(z, (40, 40))
    # axis.pcolormesh(x, y, z, cmap="YlGn")
    input_xs_t = input_xs.transpose()
    z_tensor = torch.stack(z)
    z_tensor_t = z_tensor.t()
    axis.quiver(
        input_xs_t[0],
        input_xs_t[1],
        z_tensor_t[0].detach().numpy(),
        z_tensor_t[1].detach().numpy(),
        #scale_units="xy",
        #units="xy",
        #angles="xy",
        #scale=0.001,
    )
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)

    plt.show()


def vis_path(vis, saveidx=None, slice=None):
    """ Visualize the path """
    plt.figure()
    axis = plt.gca()
    for v in vis:
        path_rec = v[1]
        goal = v[3]


        if slice is None:
            pth = list(zip(*path_rec))
        else:
            pth = list(zip(*path_rec[slice - 1:slice]))
        axis.plot(*pth, color="green")
        axis.scatter(*goal, color="red", s=250)

    axis.add_patch(plt.Rectangle((0.05, 0.05), 0.35, 0.35, fc="r", alpha=0.1))
    axis.add_patch(plt.Rectangle((-0.4, 0.05), 0.35, 0.35, fc="r", alpha=0.1))
    axis.add_patch(plt.Rectangle((0.05, -0.4), 0.35, 0.35, fc="r", alpha=0.1))
    axis.add_patch(plt.Rectangle((-0.4, -0.4), 0.35, 0.35, fc="r", alpha=0.1))

    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)
    if saveidx is None:
        plt.show()
    else:
        plt.savefig("plots/for_gifs/img_{}".format(saveidx))
        plt.close()


def get_mesh():
    input_x = torch.arange(-1, 1, 0.05)
    input_y = torch.arange(-1, 1, 0.05)

    input_xy = np.stack(np.meshgrid(input_x, input_y))
    input_xy = input_xy.reshape(2, -1)
    # input_xy = torch.tensor(input_xy).t()
    return input_xy.transpose()


def vis_heatmap(model):
    input_xs = get_mesh()
    z = [
        select_model_action(model, x)[2].mean().item() for x in input_xs
    ]
    plt.figure()
    axis = plt.gca()
    x, y = input_xs.transpose()[0], input_xs.transpose()[1]
    x = np.reshape(x, (40, 40))
    y = np.reshape(y, (40, 40))
    z = np.reshape(z, (40, 40))
    axis.pcolormesh(x, y, z, cmap="PiYG")
    #axis.imshow(z, vmin=z.min(), vmax=z.max())
    # axis.pcolormesh(x, y, z, cmap="Reds", alpha=0.5)
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)

    plt.show()


def run(model, lr, unfreeze, args):
    """Run"""
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
        model,
        args,
        num_episodes=NUM_EPISODES,
        learning_rate=lr,
        num_updates=NUM_UPDATES,
        vis=True,
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    vis_path(vis)
    # vis_heatmap(model)
    return c_reward


def main():
    """Main"""
    import matplotlib.pyplot as plt

    args = get_args()

    num_exp = NUM_EXP

    model_filename = MODEL_PATH

    m1_orig = torch.load(model_filename)
    m1 = ControllerCombinator(2, 100, 2, load_instinct=args.load_instinct)
    m1.load_state_dict(m1_orig[0].state_dict())
    experiment1_fits = [run(m1, False, m1_orig[1], args) for _ in range(num_exp)]
    experiment1_fits = list(zip(*experiment1_fits))
    with open("experiment1.list", "wb") as pckl_file1:
        pickle.dump(experiment1_fits, pckl_file1)


if __name__ == "__main__":
    main()
