#!/bin/bash
python3 general_model_exp.py --exp-ids=goodput_vs_rate --parallel --exp-name=all_transformers_vs_rate --output=res_general_vs_rate.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=all_transformers


python3 general_model_exp.py --exp-ids=goodput_vs_rate --parallel --exp-name=mixed_vs_rate --output=res_general_vs_rate.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=mixed


python3 plot_various_metrics.py --input all_transformers_vs_rate/res_general_vs_rate.tsv --general

python3 plot_various_metrics.py --input mixed_vs_rate/res_general_vs_rate.tsv --general

