"""Visualise a trained model"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from train_test_model import episode_rollout, train_maml_like
from arguments import get_args
import navigation_2d
from model import Controller, ControllerCombinator

def vis_path(path_rec, action_vec, model_info, goal):
    ''' Visualize the path '''
    fig = plt.figure()

    unzipped_info = list(zip(*model_info))
    combinator_means = unzipped_info[0]
    votes = unzipped_info[1]
    votes = list(map(lambda x: np.split(x, 2), votes))
    votes1 = list(map(lambda x: x[0], votes))
    votes2 = list(map(lambda x: x[1], votes))

    prec = list(zip(*path_rec))
    # --------------------
    # plot combinator vectors
    # --------------------
    #combinator_mean_xs, combinator_mean_ys = zip(*combinator_means)
    #axis = plt.gca()
    #axis.quiver(prec[0], prec[1], combinator_mean_xs, combinator_mean_ys,
    #            angles='xy', scale_units='xy', scale=1, color='green', headaxislength=0, headlength=0)
    
    # --------------------
    # plot module 1 vectors
    # --------------------
    #votes1_xs, votes1_ys = zip(*votes1)
    #axis = plt.gca()
    #axis.quiver(prec[0], prec[1], votes1_xs, votes1_ys,
    #            angles='xy', scale_units='xy', scale=1, color='blue', headaxislength=0, headlength=0)

    #### --------------------
    #### plot module 2 vectors
    #### --------------------
    #votes2_xs, votes2_ys = zip(*votes2)
    #axis = plt.gca()
    #axis.quiver(prec[0], prec[1], votes2_xs, votes2_ys,
    #            angles='xy', scale_units='xy', scale=1, color='red', headaxislength=0, headlength=0)

    # --------------------
    # plot path
    # --------------------
    
    plt.plot(*prec, '-o')
    plt.plot(goal[0], goal[1], 'r*')
    plt.show()

def run(model_filename):
    '''Run'''
    args = get_args()
    task_idx = 1
    #model_filename = "./trained_models/pulled_from_server/model995.pt"
    #model_filename = "./trained_models/pulled_from_server/maml_like_model_20episodes_lastGen436.pt"
    #model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model597.pt"
    #model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model977.pt"
    #model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_985.pt"
    m = torch.load(model_filename)
    #m = Controller(2, 100, 2)
    #m = ControllerCombinator(2, 2, 100, 2)
    env = navigation_2d.Navigation2DEnv()
    #env.seed(args.seed)

    ## DEBUGGING 
    #for p in m.parameters():
    #    p.data = torch.randn_like(p.data) * 10
    ###

    import numpy as np
    #c_reward, reached, _, vis = episode_rollout(module, env, 0, vis=True) 
    c_reward, reached, vis = train_maml_like(m, env, 1, args, num_episodes=20, num_updates=1, vis=True)
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    #vis_path(vis[1][:-1], vis[0], vis[2], vis[3])
    return c_reward

def main():
    '''Main'''
    import matplotlib.pyplot as plt

    model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_985.pt"
    experiment1_fits1 = [run(model_filename) for _ in range(100)]
    model_filename2 = "./trained_models/pulled_from_server/20random_goals_4modules8outputs_20episode_monolith_multiplexor/individual_999.pt"
    experiment2_fits = [run(model_filename2) for _ in range(100)]

    fig1, ax1 = plt.subplots()
    ax1.set_title('Evaluation fitness')
    ax1.boxplot([experiment1_fits1, experiment2_fits], labels=["2 outputs", "8 outputs"])
    plt.show()


    

if __name__ == '__main__':
    main()
