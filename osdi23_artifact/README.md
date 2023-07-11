# AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving

This is the artifact for the paper "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving". We are going to reproduce the main results in the paper.

## Setup the environment

Install alpa_serve package by running

```shell
pip install -e .
```
in the root folder of `mms` project.

Launch the ray runtime

```shell
ray start --head
```

Now you are ready to reproduce all the main results in the paper.

### Dataset

To get and use Azure Function Trace Dataset, read [this instruction](../alpa_serve/trace/README.md).

### Profiling results

Our algorithm relies on the profiling results, which is provided as `profiling_result.pkl` in [this issue](https://github.com/alpa-projects/mms/issues/14). If you want reproduce the profiling results yourself, plese follow [this benchmarking script](https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/benchmark_one_case_gpt_bert_inference.py) and [this conversion script](https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/gen_serving_database.py).

## End-to-end Results (Section 6.2, Figure. 11)

Generate data under `sec6_2_data/` (1 hour)

```shell
bash gen_data_sec6_2_e2e.sh
```

Currently, the script above only produces the data for the 1st, 2nd, 4th, and 5th columns in Figure. 11. The data for the 3thd and 6th columns, i.e., `S3@MAF1` and `S3@MAF2`, take a long time to run (~ 5 hours), so we provide the data in advance. If you want to reproduce the results, please uncomment the last two commands in `gen_data_sec6_2_e2e.sh`.

Plot figures under `paper_figures/`. There are four figures `goodput_vs_num_devices.pdf`, `goodput_vs_rate_scale.pdf`, `goodput_vs_cv_scale.pdf`, `goodput_vs_slo_scale.pdf` in total, each represents one row in Figure 11, respectively.

```
python3 plot_sec6_2_e2e.py --pdf
```

## Serving Very Large Models (Section 6.3, Figure. 12)

This experiment was originally done on eight p3.16xlarge AWS instances with 64 GPUs and run for over 20 hours.
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

Generate data under `sec6_4_data/` (10 min)

```
bash gen_data_sec6_4_robust.sh
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

Please refer to [this instruction](../experiments/motivation/README.md) to reproduce the results in the motivation section.