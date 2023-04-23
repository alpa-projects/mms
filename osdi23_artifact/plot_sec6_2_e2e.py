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
methodcolors = ["C2", "C1", "C0", "red"]

def plot_goodput_common(azurev1_data, azurev2_data, threshold, increasing, xlabel, output, ybottem, plot_legend=False):
    fig, axs = plt.subplots(1, len(azurev1_data) * 2)
    titles = ["S1 @ MAF1", "S2 @ MAF1", "S3 @ MAF1", "S1 @ MAF2", "S2 @ MAF2", "S3 @ MAF2"]

    for data, ax, title in zip(azurev1_data + azurev2_data, axs, titles):
        methods = list(data.keys())
        if "mp-search" in methods and "mp-search-sep" in methods:
            methods.remove("mp-search")
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
        ax.set_ylim(bottom=ybottem, top=max(y_max * 1.02, 100))
        ax.grid()

        ax.set_xlabel(xlabel, fontsize=20)
        # ax.set_ylabel("Workload Satisfaction (%)")
        if plot_legend:
            ax.set_title(title, fontsize=20)

        for i in range(len(methods)):
            if first_good[i] == 0:
                continue
            ax.axvline(first_good[i], color=methodcolors[i], linestyle=":", linewidth=4)
    
    fig.text(0.1, 0.5, "SLO Attainment (%)", va='center', rotation='vertical', fontsize=20)

    if plot_legend:
        fig.legend(curves, legends, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.2), fontsize=20)


    figure_size = (40, 4)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")



def plot_goodput_vs_num_devices(azurev1_lines, azurev2_lines, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_devices -> goodput]]
    azurev1_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev1_lines))]
    azurev2_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev2_lines))]

    for data, lines in zip(azurev1_data, azurev1_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_devices":
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_devices"], line["goodput"])
            data[policy][x] = goodput
    
    for data, lines in zip(azurev2_data, azurev2_lines):
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

    plot_goodput_common(azurev1_data, azurev2_data, threshold, True, "#devices", output, ybottom, plot_legend)


def plot_goodput_vs_num_models(azurev1_lines, azurev2_lines, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_models -> goodput]]
    azurev1_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev1_lines))]
    azurev2_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev2_lines))]

    for data, lines in zip(azurev1_data, azurev1_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_models":
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_models"], line["goodput"])
            data[policy][x] = goodput
    
    for data, lines in zip(azurev2_data, azure_v2_lines):
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

    plot_goodput_common(azurev1_data, azurev2_data, threshold, False, "#models", output, ybottom, plot_legend)


def plot_goodput_vs_slo(azurev1_lines, azurev2_lines, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[slo -> goodput]]
    azurev1_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev1_lines))]
    azurev2_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev2_lines))]

    for data, lines in zip(azurev1_data, azurev1_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_slo":
                continue

            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            data[policy][x] = goodput
    
    for data, lines in zip(azurev2_data, azure_v2_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_slo":
                continue

            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_slo.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_slo.png")

    plot_goodput_common(azurev1_data, azurev2_data, threshold, True, "SLO Scale", output, ybottom, plot_legend)


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


def plot_goodput_vs_rate_scale(azurev1_lines, azurev2_lines, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[rate_scale -> goodput]]
    azurev1_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev1_lines))]
    azurev2_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev2_lines))]

    for data, lines in zip(azurev1_data, azurev1_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_rate_scale":
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
            data[policy][x] = goodput
    
    for data, lines in zip(azurev2_data, azure_v2_lines):
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

    plot_goodput_common(azurev1_data, azurev2_data, threshold,  False, "Rate Scale", output, ybottom, plot_legend)


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



def plot_goodput_vs_cv_scale(azurev1_lines, azurev2_lines, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[cv_scale -> goodput]]
    azurev1_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev1_lines))]
    azurev2_data = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azurev2_lines))]

    for data, lines in zip(azurev1_data, azurev1_lines):
        for line in lines:
            if line["exp_name"] != "goodput_vs_cv_scale":
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
            data[policy][x] = goodput
    
    for data, lines in zip(azurev2_data, azure_v2_lines):
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

    plot_goodput_common(azurev1_data, azurev2_data, threshold, False, "CV Scale", output, ybottom, plot_legend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sec6_2_data")
    parser.add_argument("--output-dir", type=str, default="paper_figures")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    threshold = 0.99

    bert1dot3_azurev1 = args.input + "/azure_v1_1dot3b.tsv"
    bert6dot7_azurev1 = args.input + "/azure_v1_6dot7b.tsv"
    mixed_azurev1 = args.input + "/azure_v1_mixed.tsv"

    bert1dot3_azurev2 = args.input + "/azure_v2_1dot3b.tsv"
    bert6dot7_azurev2 = args.input + "/azure_v2_6dot7b.tsv"
    mixed_azurev2 = args.input + "/azure_v2_mixed.tsv"

    azure_v1_lines = [read_equal_model_case_tsv(bert1dot3_azurev1),
                      read_equal_model_case_tsv(bert6dot7_azurev1),
                      read_general_model_case_tsv(mixed_azurev1)]

    azure_v2_lines = [read_equal_model_case_tsv(bert1dot3_azurev2),
                      read_equal_model_case_tsv(bert6dot7_azurev2),
                      read_general_model_case_tsv(mixed_azurev2)]

    plot_goodput_vs_num_devices(azure_v1_lines, azure_v2_lines, threshold, args.output_dir, args.pdf, 60, True)
    plot_goodput_vs_rate_scale(azure_v1_lines, azure_v2_lines, threshold, args.output_dir, args.pdf, 60)
    plot_goodput_vs_cv_scale(azure_v1_lines, azure_v2_lines, threshold, args.output_dir, args.pdf, 40)
    plot_goodput_vs_slo(azure_v1_lines, azure_v2_lines, threshold, args.output_dir, args.pdf, 0)
