import argparse

from benchmarks.alpa.all_equal_case import read_all_equal_case_tsv


def find_max_num_models(data, num_devices, model_type, slo, policy_name, goodput):
    max_num_models = 0
    for row in data:
        if (row["num_devices"] == num_devices and
           row["model_type"] == model_type and
           row["policy_name"] == policy_name and
           abs(row["slo"] - slo) < 1e-5 and
           row["goodput"] >= goodput):
            max_num_models = max(max_num_models, row["num_models"])

    return max_num_models


def find_min_num_devices(data, model_type, num_models, slo, policy_name, goodput):
    min_num_devices = 1e10
    for row in data:
        if (row["num_models"] == num_models and
           row["model_type"] == model_type and
           row["policy_name"] == policy_name and
           row["slo"] <= slo + 1e-5 and
           row["goodput"] >= goodput - 1e-5):
            min_num_devices = min(min_num_devices, row["num_devices"])

    return min_num_devices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="res_num_models.tsv")
    parser.add_argument("--output", type=str, default="goodput.png")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    data = read_all_equal_case_tsv(args.input)

    # maximum num models one can serve
    num_devices = 8
    model_type = "bert-2.6b"
    slos = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    policy_names = ["sr-greedy", "mp-greedy-4", "mp-search"]
    goodput = 0.99

    print("- maximum #models")
    for slo in slos:
        print(f"-- slo = {slo}")
        for policy_name in policy_names:
            max_num_models = find_max_num_models(
                data, num_devices, model_type, slo, policy_name, goodput)
            print(f"    {policy_name:12} = {max_num_models}")
    print()

    # num devices required
    model_type = "bert-2.6b"
    num_models_list = [1, 2, 4, 6, 8]
    policy_names = ["sr-greedy", "mp-greedy-4", "mp-search"]
    goodput = 0.99
    slo = 0.9

    print("- min #devices")
    for num_models in num_models_list:
        print(f"-- #models = {num_models}")
        for policy_name in policy_names:
            min_num_devices = find_min_num_devices(
                data, model_type, num_models, slo, policy_name, goodput)
            print(f"    {policy_name:12} = {min_num_devices}")
