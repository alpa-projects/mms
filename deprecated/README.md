# DLSIM

## Input

Currently the simulator has no intelligence, you should prepare the [workload](./workload/README.md) and [model placement policy](./placements/README.md) first before running simulation.

## Usage

```txt
usage: simulator.py [-h] --name NAME [-n NUM_NODES] [-d NUM_DEVICES_PER_NODE] [-c MEMORY_CAPACITY] -w WORKLOAD -p PLACEMENT [--chrome_trace]

Cluster simulator for distributed DL inference tasks

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           simulation name
  -n NUM_NODES, --num_nodes NUM_NODES
                        number of nodes in the cluster
  -d NUM_DEVICES_PER_NODE, --num_devices_per_node NUM_DEVICES_PER_NODE
                        number of devices per node in the cluster
  -c MEMORY_CAPACITY, --memory_capacity MEMORY_CAPACITY
                        GPU memory capacity in GB
  -w WORKLOAD, --workload WORKLOAD
                        Workload Filename
  -p PLACEMENT, --placement PLACEMENT
                        Placement Filename
  --chrome_trace        Dump chrome trace
```

Example: `python3 simulator.py --name test_sim -n 1 -d 2 -c 16 -w skewed_workload -p placement_baseline.csv --chrome_trace`

It will print the simulation results in the console. Also, there will be a `test_sim.json` under `chrome_trace` folder. You can open `chrome://tracing` in chrome and load this json file to see the simulation results.

## TODO

- workload abstraction
  - [x] plot
  - [x] open-loop poisson generator
  - [x] workload save
  - [x] workload load
  - [ ] closed-loop workload generator
- model abstraction
  - [x] executable
  - [x] model statistics (stage latencies)
  - [ ] memory statistics (params + activation)
  - [ ] more general to read in the profile csv directly
- cluster abstraction
  - [x] device hierarchy
  - [x] submesh
  - [x] per-GPU task queue
  - [x] per-GPU clock
- placement abstraction
  - [x] placement strategy save/load
  - [x] submesh <--> executable
  - [x] inter-op executor
  - [x] intra-op executor
- simulation execution engine
  - [x] scheduler abstraction
  - [x] logging utilities
  - [x] tracing plotter
  - [ ] fix the ray overhead magic number
  - [ ] memory checker
- check with real-execution trace
  - [x] baseline trace
  - [x] inter-op only trace
  - [x] intra-op only trace
  - [ ] inter-op + intra-op trace
