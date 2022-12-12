#!/bin/bash
python3 general_model_exp.py --exp-ids=all --parallel --exp-name=all_transformers_all output=res_general_vs_all.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=all_transformers


python3 general_model_exp.py --exp-ids=all --parallel --exp-name=mixed_all --output=res_general_vs_all.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=mixed


python3 plot_various_metrics.py --input all_transformers_all/res_general_vs_all.tsv --general

python3 plot_various_metrics.py --input mixed_all/res_general_vs_all.tsv --general

