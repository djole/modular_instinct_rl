import copy
import statistics
from math import log

import torch
from gym.utils import seeding
import gym
import safety_gym

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_default_ppo
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args

ENV_NAME = "Navigation2d-v0"
NUM_PROC = 1

def train_maml_like_ppo_(
    init_model,
    args,
    learning_rate,
    num_steps=4000,
    num_updates=1,
    vis=False,
    run_idx=0,
    use_linear_lr_decay=False,
    inst_on=True,
    start_state=(0, 0)
):

    torch.set_num_threads(1)
    device = torch.device("cpu")

    envs = make_vec_envs(ENV_NAME, seeding.create_seed(None), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

    actor_critic = copy.deepcopy(init_model)
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=learning_rate,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(num_steps, NUM_PROC,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    fitnesses = []

    for j in range(num_updates):

        # if args.use_linear_lr_decay:
        #    # decrease learning rate linearly
        #    utils.update_linear_schedule(
        #        agent.optimizer, j, num_updates,
        #        agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, (final_action, _) = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], instinct_on=inst_on)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(final_action)
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms

        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device)
        print(f"fitness {fits} update {j+1}")
        if (j+1) % 1 == 0:
            print(f'----- update num {j} -----')
            print(f'action loss ---> {action_loss}')
            print(f'value loss ---> {value_loss}')
            print('-------------------------------')
        fitnesses.append(fits)

    return fitnesses[-1]


if __name__ == "__main__":
    args = get_args()

    envs = make_vec_envs(
        ENV_NAME, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )
    print("start the train function")
    init_sigma = args.init_sigma

    init_model = init_default_ppo(envs, log(init_sigma))

    start_state = (0.0, 0.0)
    cum_fitness = []
    cum_offending = []
    for g in range(20):
        #init_model = init_ppo(envs, log(init_sigma))
        #init_model.instinct = saved_model.instinct
        fitness = train_maml_like_ppo_(
            init_model,
            args,
            args.lr,
            num_steps=4000,
            num_updates=1,
            run_idx=g%4,
            inst_on=True,
            start_state=start_state
        )
        cum_fitness.append(fitness)

    print("Fitness stats")
    print(statistics.mean(cum_fitness), statistics.stdev(cum_fitness))
    print("offending stats")
    print(statistics.mean(cum_offending), statistics.stdev(cum_offending))

