import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import re
from os.path import join
import pandas as pd


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


def plot_graph(graphs, labels, color_list, x_lbl, y_lbl, title):
    for gr, clr, i in zip(graphs, color_list, range(11)):
        #if i == 2:
        #    continue
        #if len(gr) > 250:
        #    gr = gr[:250]
        #else:
        #    print(i)
        #    print(len(gr))
        #    print("----------")
        if i == 0:
            plt.plot(range(len(gr)), gr, label="MLIN", color=clr)
        if i == 5:
            plt.plot(range(len(gr)), gr, label=" no MLIN", color=clr)
        #if i == 10:
        #    plt.plot(range(len(gr)), gr, label="MLIN, no hazard", color=clr)
        else:
            plt.plot(range(len(gr)), gr, color=clr)

    #for gr, clr, lbl in zip(graphs, color_list, labels):
    #    plt.plot(range(len(gr)), gr, label=lbl, color=clr)

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():

    files_dir1 = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/models4paper/logs/"
    files_dir2 = ""
    best_vals = []
    best_activations = []
    numbers = [
        ### Get IMN fitness
        plot_concatinate_files(
            files_dir1, [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_1_part{prt}.log" for prt in [1, 2, 3]]
        ),
        plot_concatinate_files(
            files_dir1,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_2_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir1,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_3_part{prt}.log" for prt in [1, 2, 3]]
        ),
        plot_concatinate_files(
            files_dir1,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_4_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir1,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_5_part{prt}.log" for prt in [1,2]]
        ),

        # Get CTRL fitness
        plot_concatinate_files(
            files_dir1, [f"control_scaled/scaled_CTRL_1_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir1, [f"control_scaled/scaled_CTRL_2_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir1, [f"control_scaled/scaled_CTRL_3_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir1, [f"control_scaled/scaled_CTRL_4_part{prt}.log" for prt in [1, 2]]
        ),

        # Get NO HAZARD FITNESS
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/nohazard_run1_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/nohazard_run2_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/nohazard_run3_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/nohazard_run4_part{prt}.log" for prt in [1]]
        ),

        # Get NO HAZARD NO INSTINCT FITNESS
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/CTRL_nohazard_run1_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/CTRL_nohazard_run2_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir1, [f"no_hazard_CTRL/CTRL_nohazard_run3_part{prt}.log" for prt in [1]]
        ),
    ]

    numbers = list(zip(*numbers))

    mlin_data = pd.DataFrame(numbers[0][0:5])
    ctrl_data = pd.DataFrame(numbers[0][5:9])
    nohaz_data = pd.DataFrame(numbers[0][9:13])
    nohaz_ctrl_data = pd.DataFrame(numbers[0][13:16])

    data = [mlin_data, ctrl_data]
    data_nohaz = [nohaz_data, nohaz_ctrl_data]
    labels = ["Meta-learning with instincts", "Meta-learning"]
    colors = ["blue", "orange"]

    cut_limit = 220

    colors_nohaz = ["green", "cyan"]
    labels_nohaz = ["Meta-learning with instincts, no hazards", "Meta-learning, no hazards"]
    for d, lbl, c in zip(data_nohaz, labels_nohaz, colors_nohaz):
        data_mean = d.mean()[0:cut_limit]
        std = d.std()[0:cut_limit]
        std_high = data_mean + std
        std_low = data_mean - std
        plt.plot(range(len(data_mean)), data_mean, label=lbl, color=c)
        plt.fill_between(range(len(data_mean)), std_low, std_high, alpha=0.2, color=c)
        # plt.fill_between(range(len(data_mean)), d.min(), d.max(), alpha=0.2, color=c)

    for d, lbl, c in zip(data, labels, colors):
        data_mean = d.mean()[0:cut_limit]
        std = d.std()[0:cut_limit]
        std_high = data_mean + std
        std_low = data_mean - std
        plt.plot(range(len(data_mean)), data_mean, label=lbl, color=c)
        plt.fill_between(range(len(data_mean)), std_low, std_high, alpha=0.2, color=c)
        #plt.fill_between(range(len(data_mean)), d.min(), d.max(), alpha=0.2, color=c)



    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("best individual fitness")
    plt.legend(loc="lower right")
    plt.show()


    #plot_graph(
    #    numbers[0],
    #    #["evolved MLIN, lr=0.01", "no MLIN, lr=0.01, init_lr=0.005", "no MLIN, init_lr=0.0025", "no MLIN, scaled output, lr=0.01"],
    #    #["MLIN fitness", "no MLIN", "no MLIN scaled outputs (0.5)", "no MLIN scaled outputs (0.5) run 2"],
    #    #["blue", "red", "orange", "green", "teal"],
    #    ["MLIN run1", "MLIN run2", "MLIN run3", "MLIN run4", "MLIN run5"],
    #    ["blue", "blue", "blue", "blue", "blue", "orange", "orange", "orange", "orange"],
    #    "generation",
    #    "fitness",
    #    "best individual fitness",
    #)

if __name__ == "__main__":
    main()

