# Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving

This is the artifact for the paper "Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving". We are going to reproduce the main results in the paper.

## Setup the environment

### Start && Login the instance

We provide a shared AWS account for the reviewers. Click the [console login link](https://222351104556.signin.aws.amazon.com/console) and use the username and password provided in the hotcrp `Artifact Appendix` to login.

After login, click  `EC2` in the `console home` panel and then click `instances` in the left panel. You will find a `stopped` instance called `osdi23-ae` (If you find the instance in the `running` state, there may be other reviewers have already started the instance, you can skip to the next step). Right-click on the instance entry, choose `Start instance`, and wait for the instance state to become `running`.

Now you can ssh into the instance. Save the SSH keys provided in the hotcrp into a file `ae-reviewers.pem` and run the command below (you can find the public_ip by left-clicking on the instance entry and checking `Details: Public IPv4 address` in the panel showed in the bottom):

```shell
ssh -i path/to/ae-reviewers.pem ubuntu@public_ip
```

If you encounter a permission error "Permissions 0644 for 'ae-reviewers.pem' are too open", change its permission by running:

```shell
chmod 400 ae-reviewers.pem
```

And then rerun the ssh command.

After ssh into the instance, change directory into the artifact working directory:

```shell
cd /home/ubuntu/mms/osdi23_artifact
```

Launch the Ray runtime if the instance is started by yourself and no one has already launched it, or you may encounter an error saying that Ray is already running:

```shell
ray start --head
```

Now you are ready to reproduce all the main results in the paper.

### Dataset (optional)

If you are curious about how to get and use Azure Function Trace Dataset, read [this instruction](../alpa_serve/trace/README.md). For AE, they have been already setup.

### Profiling results (optional)

Our algorithm relies on the profiling results, which is provided as `profiling_result.pkl` in the current folder. If you want reproduce the profiling results, plese follow [this benchmarking script](https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/benchmark_one_case_gpt_bert_inference.py) and [this conversion script](https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/gen_serving_database.py).

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

## Stop the instance

After you finish the AE, please stop the running instance by right-click on the instance entry on AWS and choose `Stop instance`.