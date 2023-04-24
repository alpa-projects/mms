import pickle
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt
from alpa_serve.util import GB

def get_database_data(database, model_name="bert-2.6b", op=True):
    n_gpus = [1, 2, 4, 8]
    n_gpus_str = [str(x) for x in n_gpus]
    base_latency = pp_all_latency = database[model_name].para_dict[(1, 1, 1)].latency[1][0]
    pp_comm = []
    pp_uneven = []
    op_comm = []
    for n in n_gpus:
        pp_all_latency = database[model_name].para_dict[(1, 1, n)].latency[1]
        comm = max(sum(pp_all_latency) - base_latency, 0)
        uneven = max(max(pp_all_latency) * n - comm - base_latency, 0)
        pp_comm.append(comm)
        pp_uneven.append(uneven)
        if op:
            op_all_latency = database[model_name].para_dict[(1, n, 1)].latency[1]
            comm = op_all_latency[0] - base_latency / n
            op_comm.append(comm)
    return n_gpus, n_gpus_str, base_latency, pp_comm, pp_uneven, op_comm


def plot_one_database(n_gpus, n_gpus_str, base_latency, pp_comm, pp_uneven, op_comm):
    plt.figure(figsize=(3, 2.5))
    plt.grid(axis="y")
    plt.bar(n_gpus_str, [base_latency] * len(n_gpus), label="Compuation", width=0.3)
    plt.bar(n_gpus_str, pp_comm, label="Communication Overhead", bottom = [base_latency] * len(n_gpus), width=0.3)
    plt.bar(n_gpus_str, pp_uneven, label="Uneven Partition Overhead", bottom = [base_latency + x for x in pp_comm], width=0.3)
    plt.xlabel("Number of GPUs")
    plt.ylabel("Latency (s)")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"overhead_decomposition_pp.pdf")

    plt.figure(figsize=(3, 2.5))
    plt.grid(axis="y")
    plt.bar(n_gpus_str, [base_latency / n for n in n_gpus], label="Compuation", width=0.3)
    plt.bar(n_gpus_str, op_comm, label="Communication Overhead", bottom = [base_latency / n for n in n_gpus], width=0.3)
    plt.xlabel("Number of GPUs")
    plt.ylabel("Latency (s)")
    plt.ylim(0, 0.25)
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"overhead_decomposition_op.pdf")

def plot_two_databases(manual_result, dp_result, model_name="bert-2.6b", ylim=(0.2, 0.3)):
    plt.figure(figsize=(3, 2.5))
    n_gpus, n_gpus_str, base_latency, pp_comm, pp_uneven, _ = manual_result
    x = np.arange(len(n_gpus))
    width = 0.3
    ax = plt.gca()
    print(model_name, "manual", pp_comm[-1] + pp_uneven[-1])
    ax.bar(x - width/2, [base_latency] * len(n_gpus), width=0.3, color="C0", alpha=0.5)
    ax.bar(x - width/2, pp_comm, bottom = [base_latency] * len(n_gpus), width=0.3, color="C1", alpha=0.5)
    ax.bar(x - width/2, pp_uneven, bottom = [base_latency + x for x in pp_comm], width=0.3, color="C2", alpha=0.5)
    _, _, _, pp_comm, pp_uneven, _ = dp_result
    print(model_name, "dp", pp_comm[-1] + pp_uneven[-1])
    ax.bar(x + width/2, [base_latency] * len(n_gpus), width=0.3, color="C0", label="Compuation")
    ax.bar(x + width/2, pp_comm, bottom = [base_latency] * len(n_gpus), width=0.3, color="C1", label="Communication Overhead")
    ax.bar(x + width/2, pp_uneven, bottom = [base_latency + x for x in pp_comm], width=0.3, color="C2", label="Uneven Partition Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(n_gpus_str)
    ax.grid(axis="y")
    plt.ylim(*ylim)
    plt.xlabel("Number of GPUs")
    plt.ylabel("Latency (s)")
    plt.legend(loc="upper left", prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"overhead_decomposition_pp_compare_{model_name}.pdf")

if __name__ == "__main__":
    with open("profiling_result_long_sequence_manual.pkl", "rb") as f:
        manual_database = pickle.load(f)
    with open("profiling_result_long_sequence_dp.pkl", "rb") as f:
        dp_database = pickle.load(f)
    manual_result = get_database_data(manual_database, "bert-2.6b")
    dp_result = get_database_data(dp_database, "bert-2.6b")
    plot_one_database(*manual_result)
    plot_two_databases(manual_result, dp_result, "bert-2.6b", ylim=(0.2, 0.3))

    manual_result = get_database_data(manual_database, "bert-1.3b", op=False)
    dp_result = get_database_data(dp_database, "bert-1.3b", op=False)
    plot_two_databases(manual_result, dp_result, "bert-1.3b", ylim=(0.1, 0.25))
