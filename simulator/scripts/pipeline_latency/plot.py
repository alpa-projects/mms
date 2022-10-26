import matplotlib.pyplot as plt
from pprint import pprint

gpu_nums = [1, 2, 4, 8, 16, 32]
latencies = {
    "2.6B": {
        1: [0.145, 0.145, 0.144, 0.145, 0.146, 0.154],
        2: [0.187, 0.184, 0.182, 0.185, 0.197, 0.213],
        4: [0.368, 0.363, 0.362, 0.373, 0.383, 0.426],
        8: [0.736, 0.733, 0.733, 0.745, 0.78, 0.85],
        16: [1.433, 1.433, 1.444, 1.447, 1.542, 1.65]
    },
    "6.7B": {
        1: [0.233, 0.233, 0.232, 0.234, 0.243, 0.253],
        2: [0.349, 0.35, 0.35, 0.353, 0.369, 0.411],
        4: [0.7, 0.694, 0.699, 0.705, 0.739, 0.798],
        8: [1.383, 1.386, 1.39, 1.397, 1.457, 1.558],
        16: [2.823, 2.823, 2.833, 2.852, 2.959, 3.18]
    }
}

#       S    H     L    head   V
# 2.6B  1024 2560  32   32   51200
# 6.7B  1024 4096  32   32   51200

CROSS_NODE_RAY_OVERHEAD = 0.010
cross_node_commu_overhead = {
    "2.6B": 0.0015625, # 1024 * 2560 * 2 / 1024 / 1024 / (25 / 8 * 1024) # 25Gbps
    "6.7B": 0.0025,    # 1024 * 4096 * 2 / 1024 / 1024 / (25 / 8 * 1024) # 25Gbps
}

in_node_commu_overhead = {
    # "2.6B": 0.00009766, # 1024 * 2560 * 2 / 1024 / 1024 / (50 * 1024) # 50GB/s
    "2.6B": 0.0003, # profiling result
    # "6.7B": 0.00015625, # 1024 * 4096 * 2 / 1024 / 1024 / (50 * 1024) # 50GB/s
    "6.7B": 0.0004, # profiling result
}


predicted_latencies = {
    "2.6B": {
        1: [0.145], 2: [0.187], 4: [0.366], 8: [0.736], 16: [1.433]
    },
    "6.7B": {
        1: [0.231], 2: [0.349], 4: [0.7], 8: [1.383], 16: [2.823]
    }
}

for model_size in predicted_latencies:
    for bs in [1, 2, 4, 8, 16]:
        single = predicted_latencies[model_size][bs][0]
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * bs)
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * 3 * bs)
        predicted_latencies[model_size][bs].append(single + in_node_commu_overhead[model_size] * 7 * bs)
        predicted_latencies[model_size][bs].append(single + (in_node_commu_overhead[model_size] * 14 
                                                             + cross_node_commu_overhead[model_size]) * bs)
        predicted_latencies[model_size][bs].append(single + (in_node_commu_overhead[model_size] * 28
                                                             + cross_node_commu_overhead[model_size] * 3) * bs)

pprint(predicted_latencies)

def plot(model_size):
    bs_config = [1, 2, 4, 8, 16]
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