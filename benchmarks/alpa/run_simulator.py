import asyncio
from functools import partial

from alpa_serve.profiling import ProfilingResult, ParallelConfig
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import main_loop, run_event_loop
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload


async def main():
    controller = Controller()
    controller.register_model.remote(
        "a", partial(Executable, ProfilingResult.load("alpa/bert-1.3b")))

    group_id = 0
    controller.create_mesh_group_manager.remote(group_id, [1, 2])
    controller.create_replica.remote("a", group_id,
                                     [ParallelConfig(1, 1, 2)])

    w = Workload.gen_poisson("a", 0, 10, 60)
    client = Client(controller)
    client.submit_workload(w)

    await main_loop()

    client.print_stats(w, warmup=10)


if __name__ == "__main__":
    run_event_loop(main())
