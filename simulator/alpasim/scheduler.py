from typing import Dict, List

from alpasim.workload import WorkLoad
from alpasim.cluster import MeshExecutor, ScheduledTask


# TODO (Zhong Yinmin): generalize the Scheduler and write an abstract class

class FIFOScheduler:
    def __init__(self, workload: WorkLoad, meshexecutors: Dict[str, List[MeshExecutor]], model_id_to_service_name: Dict[int, str]):
        """
        @param workload: the workload to be scheduled.
        @param meshexecutors: a map from service_name to a list of its corresponding mesh_executors
        @param model_id_to_service_name: Workload only contains model_id, so it is independent of the concrete models.
                                         model_id_to_service_name are used to connect the workload with the concrete models.
        """
        self.workload = workload
        self.meshexecutors = meshexecutors
        self.requests = []
        next_meshexecutor = dict.fromkeys(meshexecutors.keys(), 0)
        
        for i, (model_id, arrive_time) in enumerate(zip(workload.model_ids, workload.arrive_times)):
            service_name = model_id_to_service_name[model_id]
            self.requests.append(ScheduledTask(i, model_id, arrive_time, arrive_time, 
                                               meshexecutors[service_name][next_meshexecutor[service_name]]))
            # round-robin
            next_meshexecutor[service_name] += 1
            next_meshexecutor[service_name] %= len(meshexecutors[service_name])
        self.completed_requests = []

    def handle_event(self):
        task = self.requests.pop(0)
        task.start_execution()
        self.completed_requests.append(task)
   
    @property
    def next_event_time(self):
        if len(self.requests) == 0:
            return float('inf')
        else:
            return self.requests[0].schedule_time
    

    