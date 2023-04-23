import argparse
import warnings
import glob
import math

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.general_model_case import read_general_model_case_tsv
from benchmarks.alpa.plot_various_metrics import show_name, method2color, method2order

linestyles = ["solid", "dashed", "dashdot", "dotted", (0, (3,5,1,5,1,5))]
methodcolors = ["C2", "C1", "C0", "red"]

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

    for i in range(len(methods)):
        if first_good[i] == 0:
            continue
        ax.axvline(first_good[i], color=methodcolors[i], linestyle=":", linewidth=4)
    
    return curves, legends


def plot_goodput(lines, threshold, folder, pdf):
    devices_data = defaultdict(lambda: defaultdict(dict)) 
    models_data = defaultdict(lambda: defaultdict(dict))
    rate_data = defaultdict(lambda: defaultdict(dict))
    cv_data = defaultdict(lambda: defaultdict(dict))
    slo_data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] == "goodput_vs_num_devices":
            policy, x, goodput = (
                line["policy_name"], line["num_devices"], line["goodput"])
            devices_data[policy][x] = goodput
        elif line["exp_name"] == "goodput_vs_num_models":
            policy, x, goodput = (
                line["policy_name"], line["num_models"], line["goodput"])
            models_data[policy][x] = goodput
        elif line["exp_name"] == "goodput_vs_rate_scale":
            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
            rate_data[policy][x] = goodput
        elif line["exp_name"] == "goodput_vs_cv_scale":
            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
            cv_data[policy][x] = goodput
        elif line["exp_name"] == "goodput_vs_slo":
            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            slo_data[policy][x] = goodput
        else:
            continue
 
    # fig, axs = plt.subplots(1, 5)
    fig, axs = plt.subplots(1, 4)

    # datas = [devices_data, models_data, rate_data, cv_data, slo_data]
    datas = [devices_data, rate_data, cv_data, slo_data]
    # xlabels = ["#devices", "#models", "Rate Scale", "CV Scale", "SLO Scale"]
    xlabels = ["#devices", "Rate Scale", "CV Scale", "SLO Scale"]
    # ybottoms = [0,0,0,0,0]
    ybottoms = [0,0,0,0]
    # increasings = [True, False, False, False, True]
    increasings = [True, False, False, True]
    for data, increasing, ax, xlabel, ybottom in zip(datas, increasings, axs, xlabels, ybottoms):
        curves, legends = plot_goodput_common(data, threshold, increasing, ax, xlabel, ybottom)
   
    fig.text(0.09, 0.5, "SLO Attainment (%)", va='center', rotation='vertical', fontsize=20)
    fig.legend(curves, legends, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=20)

    if pdf:
        output = os.path.join(folder, "robustness.pdf")
    else:
        output = os.path.join(folder, "robustness.png")

    figure_size = (30, 5)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="sec6_4_data")
    parser.add_argument("--output-dir", type=str, default="paper_figures")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    threshold = 0.99
    folder = args.input_folder
    assert os.path.isdir(args.input_folder)
    files = glob.glob(args.input_folder + "/*.tsv")
    print(f"reading files: {files}")
    results = []
    for file in files:
        results.append(read_equal_model_case_tsv(file))
    avg_results = results[0]


    def find_goodputs(results, exp_name, policy_name, value):
        ret = []
        for result in results:
            for line in result:
                if line["exp_name"] == exp_name and line["policy_name"] == policy_name:
                    if exp_name == "goodput_vs_num_devices":
                        if line["num_devices"] == value:
                            ret.append(line["goodput"])
                            break
                    elif exp_name == "goodput_vs_num_models":
                        if line["num_models"] == value:
                            ret.append(line["goodput"])
                            break
                    elif exp_name == "goodput_vs_slo":
                        if line["slo_scale"] == value:
                            ret.append(line["goodput"])
                            break
                    elif exp_name == "goodput_vs_rate_scale":
                        if math.isclose(line["arrival_process_kwargs"]["rate_scale"], value):
                            ret.append(line["goodput"])
                            break
                    elif exp_name == "goodput_vs_cv_scale":
                        if line["arrival_process_kwargs"]["cv_scale"] == value:
                            ret.append(line["goodput"])
                            break
                    else:
                        raise RuntimeError(f"Unknown exp_name: {exp_name}")
        assert len(ret) == 2
        return ret

    for line in avg_results:
        exp_name = line["exp_name"]
        policy_name = line["policy_name"]
        goodput = line["goodput"]
        # print(f"policy: {policy_name}, goodput 1 {goodput}")
        if exp_name == "goodput_vs_num_devices":
            goodputs = find_goodputs(results[1:], exp_name, policy_name, line["num_devices"])
        elif exp_name == "goodput_vs_num_models":
            goodputs = find_goodputs(results[1:], exp_name, policy_name, line["num_models"])
        elif exp_name == "goodput_vs_slo":
            goodputs = find_goodputs(results[1:], exp_name, policy_name, line["slo_scale"])
        elif exp_name == "goodput_vs_rate_scale":
            goodputs = find_goodputs(results[1:], exp_name, policy_name, line["arrival_process_kwargs"]["rate_scale"])
        elif exp_name == "goodput_vs_cv_scale":
            goodputs = find_goodputs(results[1:], exp_name, policy_name, line["arrival_process_kwargs"]["cv_scale"])
        else:
            raise RuntimeError(f"Got an unknow exp_name: {exp_name}")
        goodputs.append(goodput)
        # print(f"exp_name {exp_name}, policy {policy_name}, goodput {goodputs}")
        avg_goodput = sum(goodputs) / len(goodputs)
        line["goodput"] = avg_goodput

    plot_goodput(avg_results, threshold, args.output_dir, args.pdf)
