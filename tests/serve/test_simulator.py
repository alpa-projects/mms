"""Test alpa.serve controller."""
import asyncio
from functools import partial
import unittest

import ray

from alpa_serve.profiling import ParallelConfig, load_test_prof_result
from alpa_serve.controller import run_controller
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, Request, PoissonProcess


class EchoModel:
    def __init__(self, virtual_mesh=None):
        pass

    async def handle_request(self, request, delay=None):
        return request


class SimulatorTest(unittest.TestCase):

    async def main_test_query(self, controller):
        controller.register_model.remote("echo", EchoModel)

        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 4])
        controller.create_replica.remote("echo", group_id)

        controller.sync()

        request = Request("echo", None, None, 0, {})
        ret = controller.handle_request.remote(request)
        assert request == await ret

    def test_query(self):
        # Test the simulator
        controller = Controller()
        run_event_loop(self.main_test_query(controller))

        # Test the real system
        ray.init(address="auto")
        controller = run_controller("localhost")
        asyncio.run(self.main_test_query(controller))

    async def main_test_client(self):
        controller = Controller()
        controller.register_model.remote(
            "a", partial(Executable, load_test_prof_result("test-2GB-100ms")))

        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 2])
        controller.create_replica.remote("a", group_id,
                                         [ParallelConfig(1, 1, 2)])

        w = PoissonProcess(10).generate_workload("a", 0, 60, slo=0.15)
        client = Client(controller)
        client.submit_workload(w)

        return client, w

    def test_client(self):
        client, w = run_event_loop(self.main_test_client())
        stats = client.compute_stats(w, warmup=10)
        Workload.print_stats(stats)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimulatorTest("test_query"))
    suite.addTest(SimulatorTest("test_client"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
