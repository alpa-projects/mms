# Motivation Experiments (~1hr)

## Prepare profiling databases

Please find the profiling databases used in motivation experiments [here](https://github.com/alpa-projects/mms/issues/14#issuecomment-1521422527). There should be three databases in total:
- `profiling_result.pkl`
- `profiling_result_long_sequence_manual.pkl`
- `profiling_result_long_sequence_dp.pkl`

Please unzip the files and put them under `experiments/motivation/`.

## Two model example (Sec 3.1, Figure 2 (a)-(d))

To generate the figures:
```bash
python illustrative_example.py
```

Figures mapping:
- Figure 2 (a): `illustrative_example_1.pdf`.
- Figure 2 (b): `illustrative_example_2.pdf`.
- Figure 2 (c): `illustrative_example_3.pdf`.
- Figure 2 (d): `illustrative_example_utilization_4.pdf`.

## Changing per-GPU memory (Sec 3.2, Figure 4)

To generate the figures:
```bash
python memory_budget_vs_latency.py
```

Figures mapping:
- Figure 4 (left): `memory_budget_vs_latency_mean_latency_2.pdf`.
- Figure 4 (right): `memory_budget_vs_latency_p99_latency_2.pdf`.

## Changing arrival rates, CVs, and SLOs (Sec 3.2, Figure 5, 6, 7(a))

To generate the figures:
```bash
python changing_rate_cv_slo.py
```

Figures mapping:
- Figure 5 (left): `changing_rate_cv_slo_1.pdf`.
- Figure 5 (right): `changing_rate_cv_slo_1.5.pdf`.
- Figure 6 (left): `changing_rate_cv_slo_2.pdf`.
- Figure 6 (right): `changing_rate_cv_slo_2.5.pdf`.
- Figure 7 (a): `changing_rate_cv_slo_3.pdf`.

## Changing model parallel overhead (Sec 3.3, Figure 7(b))

To generate the figures:
```bash
python changing_pipeline_overhead.py
```

Figures mapping:
- Figure 7 (b): `changing_pipeline_overhead_1.pdf`.

## Model parallel overhead (Sec 3.3, Figure 8 & Sec 6.5, Figure 14)

To generate the figures:
```bash
python overhead_decomposition.py
```

Figures mapping:
- Figure 8 (a): `overhead_decomposition_pp.pdf`.
- Figure 8 (b): `overhead_decomposition_op.pdf`.
- Figure 14 (a): `overhead_decomposition_pp_compare_bert-1.3b.pdf`
- Figure 14 (b): `overhead_decomposition_pp_compare_bert-2.6b.pdf`


## Latency, throughput, and memory usage of model parallelism (Sec 3.3, Figure 9)

To generate the figures:
```bash
python model_parallel_latency_throughput.py
```

Figures mapping:
- Figure 9 (a): `model_parallel_latency.pdf`.
- Figure 9 (b): `model_parallel_throughput.pdf`
- Figure 9 (c): `model_parallel_memory.pdf`
