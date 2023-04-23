# Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving
This is the artifact for the paper "Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving". We are going to reproduce the main results in the paper.

## Setup the environment
TODO @yinmin
### Clean the results
Since the AWS instance is shared by the reviewers, we provide a script to clean up all the data produced earlier to ensure that each reviewer can reproduce the results by themselves.
```
bash cleanup.sh
```

## End-to-end Results (Section 6.2, Figure. 11)
Generate data under `sec6_2_data/` (1 hour)

The data for the 3thd and 6th column in Figure. 11, i.e., S3@MAF1 and S3@MAF2, take a long time to run (~ 5 hours), so we provide the data in advance. If you want to reproduce the results, uncomment the last two commands in `gen_data_sec6_2_e2e.sh`.
```
bash gen_data_sec6_2_e2e.sh
```

Plot figures under `paper_figures/`. There are four figures `goodput_vs_num_devices.pdf`, `goodput_vs_rate_scale.pdf`, `goodput_vs_cv_scale.pdf`, `goodput_vs_slo_scale.pdf` in total, each represents one row in Figure 11, respectively.
```
python3 plot_sec6_2_e2e.py --pdf
```

## Serving Very Large Models (Section 6.3, Figure. 12)
This experiment was done on 8 p3.16xlarge AWS instances with 64 GPUs and run for over 20 hours originally.
Due to the limited budget, we provide a simulated version and the accuracy of the simulator is verified in the paper.

Generate data into `sec6_3_data/large_model_exp.tsv` (5 sec)
```
bash gen_data_sec6_3_large.sh
```
Plot figure `paper_figures/large_model_exp.pdf`
```
python3 plot_sec6_3_large.py --pdf
```

## Robustness to Changing Traffic Patterns (Section 6.4, Figure. 13)
Generate data under `sec6_4_data/` (5 min)
```
bash gen_data_sec6_4_robust.py
```
Plot figure `paper_fugures/robustness.pdf`
```
python3 plot_sec6_4_robust.py --pdf
```


## Ablation Study (Section 6.5, Figure. 14)
Generate data into `sec6_5_data/ablation.tsv` (5 min)
```
bash gen_data_sec6_5_ab.sh
```
Plot figure `paper_figures/ablation.pdf`
```
python3 plot_sec6_5_ab.py --pdf
```

## Motivation results
TODO @zhuohan
