#!/bin/bash
python3 general_model_exp.py --exp-ids=goodput_vs_cv --parallel --exp-name=all_transformers_vs_cv --output=res_general_vs_cv.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=all_transformers


python3 general_model_exp.py --exp-ids=goodput_vs_cv --parallel --exp-name=mixed_vs_cv --output=res_general_vs_cv.tsv --trace-dir=/home/ubuntu/azure_v2.pkl --workload=azure_v2 --model-type=mixed


python3 plot_various_metrics.py --input all_transformers_vs_cv/res_general_vs_cv.tsv --general

python3 plot_various_metrics.py --input mixed_vs_cv/res_general_vs_cv.tsv --general

