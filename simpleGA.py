import torch
import itertools as itools
from multiprocessing import Pool
from functools import partial
from torch.distributions import Normal
import numpy as np
import time
from model import ControllerCombinator
from train_test_model import train_maml_like
import navigation_2d

import os
import copy

def _is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

NUM_PROC = 5
MAXTSK_CHLD = 10
START_LEARNING_RATE = 7e-4
D_IN, D_OUT, D_HIDDEN = 2, 2, 100

def get_population_files(load_ga_dir):

    ind_files =[name for name in os.listdir(load_ga_dir)]
    ind_files = list(map(partial(os.path.join, load_ga_dir), ind_files))

    return ind_files


class Individual:
    def __init__(self, model, device, rank):
        self.model = model
        self.device = device
        self.rank = rank
        # A set of masks that will prevent some weigths from being changed by the optimizer
        # The mask is initialized to all ones to maintain the default behavior
        self.model_plasticity_masks = []

class EA:
    def _init_model(self, deterministic, module_out, init_sigma):
        model = ControllerCombinator(D_IN, 4, D_HIDDEN, D_OUT, module_out, det=deterministic, init_std=init_sigma)
        return model

    def _compute_ranks(self, x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def _compute_centered_ranks(self, fitnesses):
        x = np.array(fitnesses)
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y.tolist()


    def __init__(self, args, device, pop_size, elite_prop):
        if pop_size < 1:
            raise ValueError("Population size has to be one or greater, otherwise this doesn't make sense")
        self.pop_size = pop_size
        self.population = [] # a list of lists/generators of model parameters
        self.selected = [] # a buffer for the selected individuals
        self.to_select = int(self.pop_size * elite_prop)
        self.fitnesses = []
        self.reached = []
        self.args = args
        
        self.sigma = 0.01
        self.sigma_decay = 0.999
        self.min_sigma = 0.001

        # if recover GA, load a list of files representing the population
        if args.load_ga:
            saved_files = get_population_files(args.load_ga_dir)

        for n in range(pop_size + self.to_select):
            if args.load_ga and n < pop_size:
                s = torch.load(saved_files[n])
                print("Load individual from {}".format(saved_files[n]))
                start_model = s[0]
            else:
                start_model = self._init_model(args.deterministic, args.module_outputs, args.init_sigma)

            ind = Individual(start_model, device, rank=n)

            if n < self.pop_size:
                self.population.append(ind)
                self.fitnesses.append(0)
                self.reached.append(0)
            else:
                self.selected.append(ind)
            print("Built {} individuals out of {}".format(n, (pop_size+self.to_select)))
    
    def ask(self):
        return self.population

    def tell(self, fitnesses):
        if len(fitnesses) != len(self.fitnesses):
            raise ValueError("Fitness array mismatch")

        fitness_list, reached_list = list(zip(*fitnesses))
        self.fitnesses = fitness_list
        self.reached = reached_list
    
    def step(self, generation_idx, args, device):
        """One step of the evolution"""
        # Sort the population by fitness and select the top
        sorted_fit_idxs = list(reversed(sorted(zip(self.fitnesses, itools.count()))))
        sorted_pop = [self.population[ix] for _, ix in sorted_fit_idxs]

        # recalculate the fitness of the elite subset and find the best individual
        elite_pop = sorted_pop[:len(self.selected)]
        #re_fit_max = float("-inf")
        #max_idx = 0
        #fitness_recalclulation_ = partial(self.fitness_calculation, num_processes=10)
        #re_fits = []

        #with Pool(processes=NUM_PROC, maxtasksperchild=MAXTSK_CHLD) as pool:
        #re_fits = map(fitness_recalclulation_, elite_pop)

        #for re_fit, (_, elite_ix) in zip(re_fits, sorted_fit_idxs):
        #    if re_fit > re_fit_max:
        #        max_idx = elite_ix
        #        re_fit_max = re_fit
        max_fitness, max_idx = sorted_fit_idxs[0]
        for cp_from, cp_to in zip(sorted_pop, self.selected):
            cp_to.model.load_state_dict(cp_from.model.state_dict())

        print("\n=============== Generation index {} ===============".format(generation_idx))
        print("best in the population ----> ", sorted_fit_idxs[0][0])
        print("best in population reached {} goals".format(self.reached[max_idx]))
        #print("best in the population after stabilization", re_fit_max)
        print("worst in the population ----> ", sorted_fit_idxs[-1][0])
        print("worst parent --------------->", sorted_fit_idxs[self.to_select-1][0])
        print("average fitness ------> ", sum(self.fitnesses)/len(self.fitnesses))
        print("===================================================\n")
            
        # next generation
        for i in range(self.pop_size):
            if i == max_idx:
                # save the best model
                state_to_save = self.population[i].model.state_dict()
                torch.save(state_to_save, r'{0}_{1}_generation{2}.dat'
                .format(self.args.save_dir,
                            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                            generation_idx))
                continue

            dart = int(torch.rand(1) * self.to_select)
            parent = self.selected[dart]
            indiv = self.population[i]
            indiv.model.load_state_dict(parent.model.state_dict())
            # apply mutation
            for p in indiv.model.parameters():
                mutation = torch.randn_like(p.data) * self.sigma
                p.data += mutation

        if self.sigma > self.min_sigma:
            self.sigma *= self.sigma_decay
        elif self.sigma < self.min_sigma:
            self.sigma = self.min_sigma
        
        return (self.population[max_idx], max_fitness)

    def fitness_calculation(self, individual, args, env, num_attempts=20):
        # fits = [episode_rollout(individual.model, args, env, rollout_index=ri, adapt=args.ep_training) for ri in range(num_attempts)]
        fits = [train_maml_like(individual.model, env, ri, args) for ri in range(num_attempts)]
        fits, reacheds, _ = list(zip(*fits))
        return sum(fits), sum(reacheds)

def save_population(args, population, best_ind, generation_idx):
    save_path = os.path.join(args.save_dir, "evolution", str(generation_idx))
    save_path_checkpoint = os.path.join(args.save_dir, "evolution", "___LAST___")
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_checkpoint):
            os.makedirs(save_path_checkpoint)
    except OSError:
        pass

    for individual in population:
        save_model = individual.model
        if args.cuda:
            save_model = copy.deepcopy(individual.model).cpu()

        torch.save(save_model, os.path.join(save_path_checkpoint, "individual_" + str(individual.rank) + ".pt"))
    
    # Save the best
    save_model = best_ind.model
    if args.cuda:
        save_model = copy.deepcopy(best_ind.model).cpu()    
    torch.save(save_model, os.path.join(save_path, "individual_" + str(generation_idx) + ".pt"))


def rollout(args, env, device, pop_size=100, elite_prop=0.1, debug=False):
    assert elite_prop < 1.0 and elite_prop > 0.0, "Elite needs to be a measure of proportion of population, 0 < elite_prop < 1"
    if debug:
        pop_size = 10
        elite_prop = 0.2

    #torch.manual_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    solver = EA(args, device, pop_size, elite_prop=elite_prop)
    fitness_list = [0 for _ in range(pop_size)]
    for iteration in range(1000):
        start_time = time.time()
        solutions = solver.ask()
        fitness_calculation_ = partial(solver.fitness_calculation, env=env, args=args)

        with Pool(processes=NUM_PROC) as pool:
            fitness_list = list(pool.map(fitness_calculation_, solutions))

        solver.tell(fitness_list)
        result, best_f = solver.step(iteration, args, device)
        # ========= Render =========
        #episode_rollout(result.model)
        #env.render_episode()
        # ==========================
        gen_time = time.time()
        save_population(args, solver.population, result, iteration)
        print("Generation: {}\n The best individual has {} as the reward".format(iteration, best_f))
        print("wall clock time == {}".format(gen_time - start_time))
    return result


def main():
    ''' main '''
    from arguments import get_args
    args = get_args()
    device = torch.device('cpu')
    args.debug = False
    env = navigation_2d.Navigation2DEnv()
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.debug:
        rollout(args, env, device, pop_size=10, elite_prop=0.1)
    else:
        rollout(args, env, device)

if __name__ == '__main__':
    main()

