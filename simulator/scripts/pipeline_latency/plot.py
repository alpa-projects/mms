import matplotlib.pyplot as plt

gpu_nums = [1, 2, 4, 8, 16, 32]
latencies = {
    "2.6B": {
        1: [0.141, 0.142, 0.142, 0.142, 0.144, 0.145],
        4: [0.366, 0.362, 0.36, 0.364, 0.378, 0.426],
        16: [1.455, 1.443, 1.433, 1.454, 1.533, 1.746],
    },
    "6.7B": {
        1: [0.231, 0.232, 0.23, 0.232, 0.236, 0.248],
        4: [0.698, 0.698, 0.699, 0.705, 0.715, 0.797],
        16: [2.810, 2.839, 2.818, 2.842, 2.959, 3.24]
    }
}

#       S    H     L    head   V
# 2.6B  1024 2560  32   32   51200
# 6.7B  1024 4096  32   32   51200
cross_node_commu_overhead = {
    "2.6B": 0.0015625 * 2.5, # 1024 * 2560 * 2 / 1024 / 1024 / (25 / 8 * 1024)
    "6.7B": 0.0025 * 2.5,    # 1024 * 4096 * 2 / 1024 / 1024 / (25 / 8 * 1024)
}

in_node_commu_overhead = {
    "2.6B": 0.00009766, # 1024 * 2560 * 2 / 1024 / 1024 / (50 * 1024)
    "6.7B": 0.00015625, # 1024 * 4096 * 2 / 1024 / 1024 / (50 * 1024)
}


predicted_latencies = {
    "2.6B": {
        1: [0.141], 4: [0.366], 16: [1.455]
    },
    "6.7B": {
        1: [0.231], 4: [0.698], 16: [2.810]
    }
}

for model_size in predicted_latencies:
    for bs in [1, 4, 16]:
        single = predicted_latencies[model_size][bs][0]
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * bs)
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * 3 * bs)
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * 7 * bs)
        predicted_latencies[model_size][bs].append(single + (in_node_commu_overhead[model_size] * 14 + cross_node_commu_overhead[model_size]) * bs)
        predicted_latencies[model_size][bs].append(single + (in_node_commu_overhead[model_size] * 28 + cross_node_commu_overhead[model_size] * 3) * bs)

print(predicted_latencies)

def plot(model_size):
    bs_config = [1, 4, 16]
    plt.figure()
    plt.title("E2E Latency v.s. #Pipeline Stages")
    plt.xlabel("#GPU (pipeline stages)")
    plt.ylabel("E2E Latency (s)")
    plt.xticks(gpu_nums)
    for bs in reversed(bs_config):
        plt.plot(gpu_nums, latencies[model_size][bs], "-o", markersize=5, label=f"BS={bs}, real")
        plt.plot(gpu_nums, predicted_latencies[model_size][bs], "--o", markersize=5, label=f"BS={bs}, calculated")
    plt.legend()
    plt.savefig(f"{model_size}.png")

plot("2.6B")
plot("6.7B")