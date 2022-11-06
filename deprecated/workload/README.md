# Workload File

Workload file is generated/saved/loaded in `alpasim/workload.py`.

## PossoinWorkload File Format

First row is metadata, which contains:

- model_num
- tot_arrival_rate
- proportions
- duration
- SLOs
- workload_name

From second row to the end are requests info, each row contains:

- model_id
- arrive_timestamp (in second)
