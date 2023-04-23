# Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving
This is the artifact for the paper "Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving". We are going to reproduce the main results in the paper.

## Setup the environment
TODO @yinmin

## End-to-end Results (Sec 6.2, Figure. 11)
TODO @yinmin

## Serving Very Large Models (Sec 6.3, Figure. 12)
This experiment was done on 8 p3.16xlarge AWS instances with 64 GPUs and run for over 20 hours.
Due to the limited budget, we provide a simulated version and the accuracy of the simulator is verified in the paper.

Generate data into `sec6_3_data/large_model_exp.tsv` (5 sec)
```
bash gen_data_sec6_3_large.sh
```
Plot figure `paper_figures/large_model_exp.pdf`
```
python3 plot_sec6_3_large.py --pdf
```

## Robustness to Changing Traffic Patterns (Sec 6.4, Figure 13)


## Ablation Study (Section 6.5, Figure 14)
Generate data into `sec6_5_data/ablation.tsv` (15 min)
```
bash gen_data_sec6_5_ab.sh
```
Plot figure `paper_figures/ablation.pdf`
```
python3 plot_sec6_5_ab.py --pdf
```

## Motivation results
TODO @zhuohan
