import csv
import json
from typing import List

from alpasim.model import Executable, model_configs

class Cluster:
    def __init__(self, num_nodes, num_devices_per_node, memory_capacity):
        """
        @param num_nodes: number of nodes in the cluster
        @param num_devices_per_node: number of GPU devices per node
        @param memory_capacity: memory capacity of each GPU device, measured in GB
        """
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.memory_capacity = memory_capacity
        self.nodes = [Node(node_id, num_devices_per_node, memory_capacity) for node_id in range(num_nodes)] 
        self.gpus = []
        for node in self.nodes:
            self.gpus.extend(node.get_all_gpus())
    
    def get_node(self, node_id):
        return self.nodes[node_id]
    
    def get_gpu(self, node_id, gpu_id):
        return self.nodes[node_id].get_gpu(gpu_id)
    
    def get_all_gpus(self):
        return self.gpus
    

class Node:
    def __init__(self, node_id, num_devices_per_node, memory_capacity):
        """
        @param node_id: index of this node in cluster
        @param num_devices_per_node: number of GPU devices in this node
        @param memory_capacity: memory capacity of each GPU device
        """
        self.node_id = node_id
        self.num_devices_per_node = num_devices_per_node
        self.gpus = [GPU(node_id, gpu_id, memory_capacity) for gpu_id in range(num_devices_per_node)]
    
    def get_gpu(self, gpu_id):
        return self.gpus[gpu_id]
    
    def get_all_gpus(self):
        return self.gpus


class GPU:
    def __init__(self, node_id, gpu_id, memory_capacity):
        """
        @param node_id: index of the node this GPU belongs to
        @param gpu_id: index of the GPU in the node it belongs to
        @param memory_capacity: GPU memory capacity
        """
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.memory_capacity = memory_capacity

        # tasks on this GPU (the first task is under execution)
        self.task_queue = []

    def receive_task(self, task):
        assert task.stage_start_time >= self.next_idle_time, \
               "GPU does not support parallel task execution"
        self.task_queue.append(task)
    
    def handle_event(self):
        assert len(self.task_queue) > 0, "GPU task queue is empty, no event to handle"
        task = self.task_queue.pop(0)
        task.exec_one_unit_work(self)

    @property
    def next_idle_time(self):
        """
        The time when the GPU was in idle, i.e. all the tasks on this GPU have finished
        """
        if len(self.task_queue) == 0:
            return 0.0
        return self.task_queue[-1].stage_finish_time
    
    @property
    def next_event_time(self):
        """
        The time when next event happens in this GPU, i.e. current running task is finished
        """
        if len(self.task_queue) == 0:
            return float('inf')
        return self.task_queue[0].stage_finish_time
   

class Mesh:
    def __init__(self, cluster: Cluster, node_ids: List[int], devices: List[List[int]]):
        assert len(node_ids) == len(devices)
        # sanity check
        for node_id, gpu_ids in zip(node_ids, devices):
            for gpu_id in gpu_ids:
                assert gpu_id < cluster.get_node(node_id).num_devices_per_node, "invalid mesh"

        self.cluster_ = cluster
        self.node_ids_ = node_ids
        self.devices_ = devices
        self.nodes_ = [self.get_node(node_id) for node_id in self.node_ids_]
        self.gpus_ = []
        for node_id, gpu_ids in zip(self.node_ids_, self.devices_):
            self.gpus_.extend([self.cluster_.get_gpu(node_id, gpu_id) for gpu_id in gpu_ids])
 
    def get_node(self, node_id):
        assert node_id in self.node_ids_, f"Mesh does not contain node {node_id}"
        return self.cluster_.get_node(node_id)
    
    def get_gpu(self, node_id, gpu_id):
        assert node_id in self.node_ids_ and gpu_id in self.devices_[node_id]
        return self.cluster_.get_gpu(node_id, gpu_id)
    
    def get_all_nodes(self):
        return self.nodes_
         
    def get_all_gpus(self):
        return self.gpus_
    
    @property
    def num_nodes(self):
        return len(self.node_ids_)
    
    @property
    def num_devices_per_node(self):
        return len(self.devices_[0])
    
    @property
    def num_devices(self):
        return self.num_nodes * self.num_devices_per_node
    
    @property
    def node_ids(self):
        return self.node_ids_
    
    @property
    def devices(self):
        return self.devices_

    @property
    def shape(self):
        return (self.num_nodes, self.num_devices_per_node)
    
    @property
    def next_idle_time(self):
        """
        Since the mesh is a collection of GPUs, the next idle time is 
        the maximum of all the next idle times of the GPUs it contains.
        """
        return max([gpu.next_idle_time for gpu in self.get_all_gpus()])
    
    def __str__(self):
        ret = ""
        for node_id in self.node_ids_:
            ret += f"Node {node_id}: {self.devices[node_id]}\n"
        return ret

class MeshGroup:
    def __init__(self, meshes: List[Mesh]):
        self.meshes = meshes
    
    def get_mesh(self, stage_num):
        return self.meshes[stage_num]
    
    @property
    def num_stage(self):
        return len(self.meshes)
    
    @property
    def stage_shapes(self):
        return [mesh.shape for mesh in self.meshes]
    
    @property
    def host_lists(self):    
        return [mesh.node_ids for mesh in self.meshes]
    
    @property
    def devices_lists(self):
        return [mesh.devices for mesh in self.meshes]
    
    @property
    def next_idle_time(self):
        return self.meshes[0].next_idle_time

    def __str__(self):
        ret = ""
        for i, mesh in enumerate(self.meshes):
            ret += f"Stage {i}:\n"
            for node_id in mesh.node_ids:
                ret += f"  Node {node_id}: {mesh.devices[node_id]}\n"
        return ret


class ScheduledTask:
    """
    A scheduled task is either waiting for scheduling or in the task_queues of all the gpus in one pipeline stage
    A scheduled task will contain all the execution statistics after simulation, including:
        - model_id: the id of the model
        - arrive_time: when the request arrive
        - schedule_time: when the request get scheduled (decided by the scheduler)
        - stage_execution_info: [
                                        [(s0_g0_start, s0_g0_end, node_id, gpu_id), (s0_g1_start, s0_g1_end, node_id, gpu_id), ...]
                                        [(s1_g0_start, s1_g0_end, node_id, gpu_id), (s1_g1_start, s1_g1_end, node_id, gpu_id), ...]
                                        ...
                                      ]
            where s0 represents the stage 0 and g0 represents gpu 0. This list is used to generate chrome trace.
        - finish_time: when the request finishes execution
    Other attributes are used for simulation:
        - meshexecutor: which meshexecutor the task is sent to
        - current_stage: which pipeline stage this task is running in
    """
    def __init__(self, request_id, model_id, arrive_time, schedule_time, SLO, meshexecutor):
        # set by scheduler
        self.model_id = model_id
        self.request_id = request_id
        self.arrive_time = arrive_time
        self.schedule_time = schedule_time
        self.SLO = SLO
        self.meshexecutor = meshexecutor
        self.num_stage = meshexecutor.executable.num_stage
        self.unit_works_per_stage = [num_node * num_devices for num_node, num_devices in meshexecutor.executable.stage_shapes]
        # set by simulator
        self.stage_execution_info = [[] for _ in range(self.num_stage)]
        self.finish_time = None
        # internal state for simulation
        self.current_stage = 0
        self.stage_start_time = None  # set by meshexecutor.receive_task()
        self.stage_finish_time = None # set by meshexecutor.receive_task()
    
    def execution_latency(self):
        return self.meshexecutor.executable.stage_latencies[self.current_stage]
    
    def start_execution(self):
        """ 
        Start the execution of this task, i.e. send this task to the mesh of first stage.
        """
        self.meshexecutor.receive_task(self, 0, self.schedule_time)
    
    def exec_one_unit_work(self, gpu: GPU):
        """
        Finish the work on one gpu in current stage.
        If all the works in current stage are done, send this task to the gpus on next stage.

        """
        self.unit_works_per_stage[self.current_stage] -= 1
        self.stage_execution_info[self.current_stage].append((self.stage_start_time, \
                                                self.stage_finish_time, gpu.node_id, gpu.gpu_id))
        if self.unit_works_per_stage[self.current_stage] == 0:
            # finish current stage
            self.current_stage += 1
            if self.current_stage == self.num_stage:
                # finish the last stage
                self.finish_time = self.stage_finish_time
            else:
                # for pipeline parallel, we need to send this task to the mesh of next stage
                self.meshexecutor.receive_task(self, self.current_stage, self.stage_finish_time)

class MeshExecutor:
    def __init__(self, service_name: str, meshgroup: MeshGroup, executable: Executable):
        # check executable defines the same mesh shapes
        meshgroup.num_stage == executable.num_stage 
        for s1, s2 in zip(meshgroup.stage_shapes, executable.stage_shapes):
            assert s1 == s2, "executable does not match with meshgroup shape"
        
        self.service_name = service_name
        self.meshgroup = meshgroup
        self.executable = executable
    
    @property
    def next_idle_time(self):
        return self.meshgroup.next_idle_time
    
    def receive_task(self, task: ScheduledTask, stage_num: int, timestamp: float):
        """ Receive a task and assign it to all the GPUs in the mesh.

        Simulation related: This function sets the stage_start_time of the task 
                            to max(receive_timestamp, mesh.next_idle_time + RAY_OVERHEAD of the mesh)

        @param task: Received task.
        @param stage_num: The stage number of the mesh to receive the task.
        @param timestamp: When the task was received.

        """
        mesh = self.meshgroup.get_mesh(stage_num)
        task.stage_start_time = max(mesh.next_idle_time, timestamp)
        # If the mesh is not idle, add Ray overhead 
        # TODO: fix this magic number
        if mesh.next_idle_time > 0:
            if task.num_stage > 1:
                task.stage_start_time += 0.003
            else:
                task.stage_start_time += 0.004

        task.stage_finish_time = task.stage_start_time + task.execution_latency()
        for gpu in mesh.get_all_gpus():
            gpu.receive_task(task)
    
    def json_dump(self):
        return {
            "service_name": self.service_name,
            "model_name": self.executable.model_name,
            "executable_name": self.executable.executable_name,
            "mesh_group": [
                {
                    "node_ids": node_ids,
                    "devices": devices
                }
                for node_ids, devices in zip(self.meshgroup.host_lists, self.meshgroup.devices_lists)
            ]
        }
    
def save_meshexecutors(meshexecutors: List[MeshExecutor], filename: str):
    """ Save all the meshexecutors into a json file.

    File format: Each element in the List defines a meshexecutor (a wrapper around model executable and meshgroup).

    There are three key-value pairs in each element:

    - service_name: the name of the serving service (different services may use the same model)
    - model_name: defined in model_configs in model.py
    - executable_name: defined in model_configs in model.py
    - mesh_group: a list of meshes which defines the placement of the model executable, each mesh contains:
        - node_ids: list of node ids
        - devices: list of device list, the length must equal the length of node_ids

    """
    with open(filename, 'w') as placement_file:
        dump = [meshexecutor.json_dump() for meshexecutor in meshexecutors]
        json.dump(dump, placement_file)

def load_meshexecutors(filename: str, cluster: Cluster):
    """ Load all the meshexecutors from json file.

    The File format is defined in save_meshexecutors.

    @param filename: the file to load
    @param cluster: the cluster which all the meshes belong to

    @return: a map from service_name to a list of meshexecutors
    """
    meshexecutors = {}
    with open(filename, 'r') as placement_file:
        placements = json.load(placement_file)
        for placement in placements:
            meshes = []
            for mesh_info in placement["mesh_group"]:
                mesh = Mesh(cluster, mesh_info["node_ids"], mesh_info["devices"])
                meshes.append(mesh)
            meshgroup = MeshGroup(meshes)
            executable = model_configs[placement["model_name"]][placement["executable_name"]]
            if placement["service_name"] not in meshexecutors:
                meshexecutors[placement["service_name"]] = []
            meshexecutors[placement["service_name"]].append(MeshExecutor(placement["service_name"], meshgroup, executable))
    return meshexecutors

