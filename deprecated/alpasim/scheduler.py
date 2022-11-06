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
        self.model_id_to_service_name = model_id_to_service_name
        self.tasks = list(iter(self.workload))
        self.scheduled_tasks = []
        self.completed_tasks = []

    def handle_event(self):
        task = self.tasks.pop(0)
        service_name = self.model_id_to_service_name[task.model_id]
        # choose the meshexecutor with shortest queue
        designated_meshexecutor = min(self.meshexecutors[service_name], key=lambda m: m.next_idle_time)
        scheduled_task = ScheduledTask(task.task_id, task.model_id, task.arrive_time, 
                                       task.arrive_time, task.SLO, designated_meshexecutor)
        self.scheduled_tasks.append(scheduled_task)
        scheduled_task.start_execution()
        self.completed_tasks.append(scheduled_task)
   
    @property
    def next_event_time(self):
        if len(self.tasks) == 0:
            return float('inf')
        else:
            # FIFO scheduler schedules tasks in the order of arrival
            return self.tasks[0].arrive_time
    

    