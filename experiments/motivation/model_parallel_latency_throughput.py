import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_latency_and_throughput(all_latency):
    latency = sum(all_latency)
    throughput = 1 / max(all_latency)
    return latency, throughput

def plot_database(database, model_name="bert-6.7b"):
    n_gpus = [1, 2, 4, 8]
    pp_latency = []
    pp_throughput = []
    pp_memory = []
    op_latency = []
    op_throughput = []
    op_memory = []
    dp_latency = []
    dp_throughput = []
    dp_memory = []
    for n in n_gpus:
        pp_all_latency = database[model_name][(1, 1, n)].latency
        pp_latency.append(get_latency_and_throughput(pp_all_latency)[0])
        pp_throughput.append(get_latency_and_throughput(pp_all_latency)[1])
        pp_memory.append(sum(database[model_name][(1, 1, n)].weight_mem))
        op_all_latency = database[model_name][(1, n, 1)].latency
        op_latency.append(get_latency_and_throughput(op_all_latency)[0])
        op_memory.append(sum(database[model_name][(1, n, 1)].weight_mem))
        op_throughput.append(get_latency_and_throughput(op_all_latency)[1])
        dp_all_latency = database[model_name][(1, 1, 1)].latency
        dp_latency.append(dp_all_latency)
        dp_throughput.append(n / max(dp_all_latency))
        dp_memory.append(sum(database[model_name][(1, 1, 1)].weight_mem) * n)

    plt.figure()
    plt.plot(n_gpus, pp_latency, label="PP")
    plt.plot(n_gpus, op_latency, label="OP")
    plt.plot(n_gpus, dp_latency, label="DP")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"model_parallel_latency.pdf")

    plt.figure()
    plt.plot(n_gpus, pp_throughput, label="PP")
    plt.plot(n_gpus, op_throughput, label="OP")
    plt.plot(n_gpus, dp_throughput, label="DP")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Throughput (req/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"model_parallel_throughput.pdf")

    plt.figure()
    plt.plot(n_gpus, pp_memory, label="PP")
    plt.plot(n_gpus, op_memory, label="OP")
    plt.plot(n_gpus, dp_memory, label="DP")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Memory (Bytes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"model_parallel_memory.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database_filename", type=str, default="profiling_result.pkl")
    args = parser.parse_args()
    with open(args.database_filename, "rb") as f:
        database = pickle.load(f)

    plot_database(database)
