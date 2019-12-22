import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import re
from os.path import join

def main():
    best_in_pop_ptrn = re.compile("best in the population ---->")
    stabilize_ptrn = re.compile("best in the population after stabilization")
    float_ptrn = re.compile("-*\d+.\d*")
    file_dir = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/4goals_dist2nogo/logs"
    prs_file = "EVOLUTION_sym_smallpop_part2.log"
    src_path = join(file_dir, prs_file)

    best_val_generation = []
    besta_stabile_generation = []
    with open(src_path, "r") as src_file:
        for log_line in src_file:
            best_line = best_in_pop_ptrn.search(log_line)
            if best_line is not None:
                val = float_ptrn.findall(log_line)[0]
                best_val_generation.append(float(val))
            best_stabile_line = stabilize_ptrn.search(log_line)
            if best_stabile_line is not None:
                val = float_ptrn.findall(log_line)[0]
                besta_stabile_generation.append(float(val))


    print(best_val_generation.index(max(best_val_generation)))
    plt.plot(range(len(best_val_generation)), best_val_generation)
    plt.plot(range(len(besta_stabile_generation)), besta_stabile_generation, '--')
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(prs_file)
    #plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()