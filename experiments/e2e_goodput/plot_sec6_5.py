import argparse
import warnings

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.plot_various_metrics import show_name, method2color, method2order

linestyles = ["solid", "dashed", "dashdot", "dotted", (0, (3,5,1,5,1,5)), (0, (3,10,1,10,1,10))]
methodcolors = ["C2", "C1", "C0", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

def plot_goodput_common(data, threshold, increasing, ax, xlabel, ybottom):
    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    first_good = []
    x_max = 0
    y_max = 0
    for i, method in enumerate(methods):
        curve = data[method]
        xs_, ys_ = zip(*curve.items())
        xs = [x for x, _ in sorted(zip(xs_, ys_))]
        ys = [y for _, y in sorted(zip(xs_, ys_))]
        ys = np.array(ys) * 100
        curve = ax.plot(xs, ys, color=methodcolors[i], marker='.', linestyle=linestyles[i], linewidth=4, markersize=15)
        curves.append(curve[0])
        legends.append(show_name(method))

        if increasing:
            iterator = range(len(xs))
        else:
            iterator = reversed(range(len(xs)))

        found = False
        for i in iterator:
            if ys[i] >= threshold * 100:
                first_good.append(xs[i])
                found = True
                break
        if not found:
            first_good.append(0)

        x_max = max(x_max, *xs)
        y_max = max(y_max, *ys)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_ylim(bottom=ybottom, top=max(y_max * 1.02, 100))
    ax.set_xlabel(xlabel, fontsize=20)
    ax.grid()
    ax.legend(curves, legends, fontsize=20)

    for i in range(len(methods)):
        if first_good[i] == 0:
            continue
        ax.axvline(first_good[i], color=methodcolors[i], linestyle=":", linewidth=4)
    
    return curves, legends


def plot_goodput(bs_lines, batching_lines, threshold, folder, pdf):
    bs_data = defaultdict(lambda: defaultdict(dict)) 
    batching_data = defaultdict(lambda: defaultdict(dict))

    for line in bs_lines:
        if line["exp_name"] == "goodput_vs_slo":
            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            bs_data[policy][x] = goodput
        else:
            continue
    
    for line in batching_lines:
        if line["exp_name"] == "goodput_vs_slo":
            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            batching_data[policy][x] = goodput
        else:
            continue
 
    # fig, axs = plt.subplots(1, 5)
    fig, axs = plt.subplots(1, 2)

    datas = [bs_data, batching_data]
    xlabels = ["SLO Scale", "SLO Scale"]
    ybottoms = [0,0,0,0]
    increasings = [True, True]
    for data, increasing, ax, xlabel, ybottom in zip(datas, increasings, axs, xlabels, ybottoms):
        curves, legends = plot_goodput_common(data, threshold, increasing, ax, xlabel, ybottom)
   
    fig.text(0.05, 0.5, "SLO Attainment (%)", va='center', rotation='vertical', fontsize=20)
    # fig.legend(reversed(curves), reversed(legends), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=20)

    if pdf:
        output = os.path.join(folder, "batching.pdf")
    else:
        output = os.path.join(folder, "batching.png")

    figure_size = (15, 5)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./paper_figures")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    threshold = 0.99

    bs_lines = read_equal_model_case_tsv(args.input + "/res_batchsize.tsv")
    batching_lines = read_equal_model_case_tsv(args.input + "/res_batching.tsv")


    plot_goodput(bs_lines, batching_lines, threshold, args.output_dir, args.pdf)
