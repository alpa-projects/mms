import argparse
from collections import defaultdict

import numpy as np

def read_data(filename):
    rows = []

    for line in open(args.input):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        exp_name, num_devices, mem_budget, model_type, num_models, per_model_rate, per_model_cv, slo, duration, policy_name, placement, goodput, mode = line.split("\t")

        num_devices = int(num_devices)
        num_models = int(num_models)
        slo = float(slo)
        duration = float(duration)
        goodput = float(goodput)

        values = locals()
        row = {
            key: values[key]
            for key in 
            ["exp_name", 
             "num_devices", "mem_budget", "model_type", "num_models",
             "per_model_rate", "per_model_cv", "slo", "duration", "policy_name",
             "placement", "goodput", "mode"]
        }
        rows.append(row)

    return rows


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
    parser.add_argument("--input", type=str, default="res_all_equal.tsv")
    parser.add_argument("--output", type=str, default="goodput.png")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    data = read_data(args.input)

    # maximum num models one can serve
    print("----- maximum #models -----")
    num_devices = 8
    model_type = "bert-2.6b"
    slos = [0.6, 0.8, 1.0, 2.0]
    policy_names = ["sr-greedy", "mp-greedy-4"]
    goodput = 0.99

    for slo in slos:
        for policy_name in policy_names:
            max_num_models = find_max_num_models(
                data, num_devices, model_type, slo, policy_name, goodput)
            print(slo, policy_name, max_num_models)
    print("---------------------------")

    # num devices required
    print("----- min #devices -----")
    model_type = "bert-2.6b"
    num_models_list = [1, 2, 4, 6, 8]
    policy_names = ["sr-greedy", "mp-greedy-4"]
    goodput = 0.99
    slo = 0.9

    for num_models in num_models_list:
        for policy_name in policy_names:
            min_num_devices = find_min_num_devices(
                data, model_type, num_models, slo, policy_name, goodput)
            print(num_models, policy_name, min_num_devices)
    print("------------------------")
