import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import re
from os.path import join


def plot_concatinate_files(files_dir, log_file_list):
    best_in_pop_ptrn = re.compile("best in the population ---->")
    stabilize_ptrn = re.compile("best in the population after stabilization")
    activation_ptrn = re.compile("activation average")
    float_ptrn = re.compile("-*\d+.\d*")

    best_val_generation = []
    best_stabile_generation = []
    best_activation_generation = []

    for log_file in log_file_list:
        src_path = join(files_dir, log_file)

        with open(src_path, "r") as src_file:
            for log_line in src_file:
                best_line = best_in_pop_ptrn.search(log_line)
                if best_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_val_generation.append(float(val))

                best_stabile_line = stabilize_ptrn.search(log_line)
                if best_stabile_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_stabile_generation.append(float(val))

                best_activation_line = activation_ptrn.search(log_line)
                if best_activation_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_activation_generation.append(float(val))

    return best_val_generation, best_stabile_generation, best_activation_generation


def plot_graph(graphs, labels, x_lbl, y_lbl, title):
    for gr, lbl in zip(graphs, labels):
        plt.plot(range(len(gr)), gr, label=lbl)

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():

    files_dir = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/largeSmallExploration/logs"

    best_vals = []
    best_activations = []
    numbers = [
        plot_concatinate_files(
            files_dir, [f"EVOLUTION_lidar_largeExp_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir,
            [f"EVOLUTION_lidar_largeExpGoalReward_part{prt}.log" for prt in [1, 2]],
        ),
        plot_concatinate_files(
            files_dir, [f"EVOLUTION_lidar_smallExp_part{prt}.log" for prt in [1, 2]]
        ),
    ]

    numbers = list(zip(*numbers))

    plot_graph(
        numbers[0],
        ["large exploration", "large exp + reward for finish", "small exploration"],
        "generation",
        "fitness",
        "best individual fitness",
    )

    plot_graph(
        numbers[2],
        ["large exploration", "large exp + reward for finish", "small exploration"],
        "generation",
        "average activation",
        "best individual average lifetime activation",
    )


if __name__ == "__main__":
    main()
