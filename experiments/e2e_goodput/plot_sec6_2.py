import argparse
import warnings

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.general_model_case import read_general_model_case_tsv
from benchmarks.alpa.plot_various_metrics import show_name, method2color, method2order

linestyles = ["solid", "dashed", "dashdot", "dotted"]

def plot_goodput_common(modelsets_data, threshold, increasing, xlabel, output):
    if len(modelsets_data) == 0:
        print(f"No data to draw for {output}. Skipped.")
        return

    fig, axs = plt.subplots(1, len(modelsets_data))

    for data, ax in zip(modelsets_data, axs):
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
            curve = ax.plot(xs, ys, color=method2color(method), marker='*', linestyle=linestyles[i], linewidth=4, markersize=15)
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
        ax.set_ylim(bottom=40, top=max(y_max * 1.02, 100))
        ax.grid()

        ax.legend(curves, legends, fontsize=20, loc="lower left")
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel("Workload Satisfaction (%)")
        # ax.set_title(title)

        for i in range(len(methods)):
            if first_good[i] == 0:
                continue
            ax.axvline(first_good[i], color=method2color(methods[i]), linestyle=":", linewidth=4)
    
    fig.text(0.5, -0.06, xlabel, ha='center', fontsize=20)
    fig.text(0.07, 0.5, "Workload Satisfaction (%)", va='center', rotation='vertical', fontsize=20)



    figure_size = (20, 4)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")



def plot_goodput_vs_num_devices(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[num_devices -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_devices":
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_devices"], line["goodput"])
            data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_num_devices.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_num_devices.png")

    plot_goodput_common(modelsets_data, threshold, True, "#devices", output)


def plot_goodput_vs_num_models(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[num_models -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_models":
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_models"], line["goodput"])
            data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_num_models.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_num_models.png")

    plot_goodput_common(modelsets_data, threshold, False, "#models", output)


def plot_goodput_vs_slo(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[slo -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_slo":
                continue

            policy, x, goodput =  (
                line["policy_name"], line["slo_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_slo.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_slo.png")

    plot_goodput_common(modelsets_data, threshold, True, "SLO Scale", output)


def plot_goodput_vs_rate(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[rate -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_rate":
                continue

            policy, x, goodput = (
                line["policy_name"], line["total_rate"], line["goodput"])
            data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_rate.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_rate.png")

    plot_goodput_common(modelsets_data, threshold, False, "Rate(r/s)", output)


def plot_goodput_vs_rate_scale(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[rate_scale -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_rate_scale":
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_rate_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_rate_scale.png")

    plot_goodput_common(modelsets_data, threshold, False, "Rate Scale", output)


def plot_goodput_vs_cv(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[cv -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_cv":
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_cv.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_cv.png")


    plot_goodput_common(modelsets_data, threshold, False, "CV", output)



def plot_goodput_vs_cv_scale(modelsets_lines, threshold, folder, pdf):
    # Dict[policy -> Dict[cv_scale -> goodput]]
    modelsets_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(modelsets_lines))]

    for data, lines in zip(modelsets_data, modelsets_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_cv_scale":
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_cv_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_cv_scale.png")

    plot_goodput_common(modelsets_data, threshold, False, "CV Scale", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-small-input", type=str, required=True)
    parser.add_argument("--bert-large-input", type=str, required=True)
    parser.add_argument("--mixed-input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="paper_figures")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    threshold = 0.99

    bert_small_lines = read_equal_model_case_tsv(args.bert_small_input)
    bert_large_lines = read_equal_model_case_tsv(args.bert_large_input)
    mixed_lines = read_general_model_case_tsv(args.mixed_input)

    plot_goodput_vs_num_devices([bert_small_lines, bert_large_lines, mixed_lines], threshold, args.output_dir, args.pdf)
    plot_goodput_vs_num_models([bert_small_lines, bert_large_lines, mixed_lines], threshold, args.output_dir, args.pdf)
    plot_goodput_vs_slo([bert_small_lines, bert_large_lines, mixed_lines], threshold, args.output_dir, args.pdf)
    plot_goodput_vs_rate_scale([bert_small_lines, bert_large_lines, mixed_lines], threshold, args.output_dir, args.pdf)
    plot_goodput_vs_cv_scale([bert_small_lines, bert_large_lines, mixed_lines], threshold, args.output_dir, args.pdf)
