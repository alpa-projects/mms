"""Test alpa.serve controller."""
import asyncio
from functools import partial
import unittest

from alpa_serve.profiling import ParallelConfig, ProfilingResult
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, Request


class EchoModel:
    def __init__(self, virtual_mesh):
        pass

    async def handle_request(self, request):
        return request


class SimulatorTest(unittest.TestCase):

    async def main_test_query(self):
        controller = Controller()

        controller.register_model.remote("echo", EchoModel)

        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 4])
        controller.create_replica.remote("echo", group_id)

        request = Request("echo", None, None)
        ret = controller.handle_request.remote(request)
        assert request == await ret

    def test_query(self):
        run_event_loop(self.main_test_query())

    async def main_test_client(self):
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

        return client, w

    def test_client(self):
        client, w = run_event_loop(self.main_test_client())
        client.print_stats(w, warmup=10)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SimulatorTest("test_query"))
    suite.addTest(SimulatorTest("test_client"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
