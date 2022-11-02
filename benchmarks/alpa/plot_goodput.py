import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


show_name_dict = {
    "sr": "SelectiveReplication",
    "mp": "Model Parallelism (ours)",
}

def show_name(name):
    return show_name_dict.get(name, name)


method2color_dict = {
    "mp": "C1",
    "sr": "C0",
}

def method2color(name):
    return method2color_dict[name]


method_order_list = [
    "sr", "mp"
]

def method2order(name):
    return method_order_list.index(name)


def read_data(filename):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in open(filename):
        exp_name, policy, slo, goodput = line.split("\t")
        data[policy][float(slo)] = float(goodput)

    return data


def plot_goodput_vs_slo(data, output, show):
    fig, ax = plt.subplots()
    figure_size = (4, 4)

    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    x_max = 0
    for method in methods:
        curve = data[method]
        xs, ys = zip(*curve.items())
        curve = ax.plot(xs, ys, color=method2color(method), marker='*')
        curves.append(curve[0])
        legends.append(show_name(method))

        x_max = max(x_max, *xs)


    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=x_max * 1.05)
    ax.set_ylabel("Goodput")
    ax.set_xlabel("SLO (ms)")
    ax.legend(curves, legends)

    if show:
        plt.show()

    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="res_goodput.tsv")
    parser.add_argument("--output", type=str, default="goodput.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = read_data(args.input)
    plot_goodput_vs_slo(data, args.output, args.show)
