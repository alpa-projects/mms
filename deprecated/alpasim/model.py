from typing import List

class Executable:
    def __init__(self, model_name: str, executable_name: str, stage_shapes: List[tuple], stage_latencies: float):
        """
        @param executable_name: used in the model_configs to index
        @param stage_shapes: list of tuple which contains the shape of submesh for each stage,
                             each tuple represents (# of nodes, # GPU per node)
        @param stage_latencies: latency for each stage
        """
        assert len(stage_shapes) == len(stage_latencies) and len(stage_shapes) > 0
        self.model_name = model_name 
        self.executable_name = executable_name
        self.stage_shapes = stage_shapes
        self.stage_latencies = stage_latencies
        self.num_stage = len(stage_shapes)
    
    def __str__(self):
        ret = f"{self.executable_name}:\n"
        for idx, (submesh, latency) in enumerate(zip(self.stage_shapes, self.stage_latencies)):
            ret += f"stage {idx}: {submesh} => {latency:.5f}s\n"
        return ret

model_configs = {
    # Benchmarked on AWS P3 instances with Tesla V100
    "Bert_125M": {
        "Bert_125M_1_device": Executable("Bert_125M", "Bert_125M_1_device", [(1,1)], [0.01893]),
        "Bert_125M_2_device_intra": Executable("Bert_125M", "Bert_125M_2_device_intra", [(1,2)], [0.0139]),
        "Bert_125M_2_device_inter": Executable("Bert_125M", "Bert_125M_2_device_inter", [(1,1),(1,1)], [0.010, 0.010]),
    },
    "Bert_2.6B": {
        "Bert_2.6B_1_device": Executable("Bert_2.6B", "Bert_2.6B_1_device", [(1,1)], [0.14520]),
        "Bert_2.6B_2_device_intra": Executable("Bert_2.6B", "Bert_2.6B_2_device_intra", [(1,2)], [0.0985]),
        "Bert_2.6B_2_device_inter": Executable("Bert_2.6B", "Bert_2.6B_2_device_inter", [(1,1),(1,1)], [0.0730, 0.0734]),
        "Bert_2.6B_4_device_inter": Executable("Bert_2.6B", "Bert_2.6B_4_device_inter", [(1,1),(1,1),(1,1),(1,1)], [0.037, 0.037, 0.037, 0.037]),
    }
}


if __name__ == '__main__':
    for model_name, execs in model_configs.items():
        for exec in execs:
            print(exec)
