#!/bin/bash
python3 general_model_exp.py --exp-ids=all --parallel --exp-name=ablation_general_synthetic_bert_all --output=res_general_vs_all.tsv --model-type=all_transformers

python3 general_model_exp.py --exp-ids=all --parallel --exp-name=ablation_general_synthetic_mixed_all --output=res_general_vs_all.tsv --model-type=mixed

python3 plot_various_metrics.py --input ablation_general_synthetic_bert_all/res_general_vs_all.tsv --general --synthetic

python3 plot_various_metrics.py --input ablation_general_synthetic_mixed_all/res_general_vs_all.tsv --general --synthetic

