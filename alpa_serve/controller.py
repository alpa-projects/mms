#pylint: disable=missing-class-docstring, raise-missing-from
"""Central controller"""
import asyncio
from collections import defaultdict
import dataclasses
import logging
import math
import os
import pickle
import socket
import time
from typing import Callable, List, Dict, Optional, Tuple, Any, Union

import numpy as np
import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from starlette.datastructures import QueryParams
from starlette.middleware.cors import CORSMiddleware
import uvicorn

from alpa_serve.http_util import (HTTPRequestWrapper, receive_http_body,
                                  Response, set_socket_reuse_port, ASGIHandler,
                                  build_starlette_request, new_port,
                                  RelayException, make_error_response)
from alpa_serve.util import build_logger, add_sync_method, to_str_round

logger = logging.getLogger(__file__)

CONTROLLER_NAME = "controller"
#SOCKET_REUSE_PORT_ENABLED = (os.environ.get("SERVE_SOCKET_REUSE_PORT_ENABLED",
#                                            "1") == "1")
SOCKET_REUSE_PORT_ENABLED = False


@dataclasses.dataclass
class CreateInfo:
    model_def: Any
    init_args: Optional[List] = None
    init_kwargs: Optional[Dict] = None

    def append_init_args(self,
                         init_args: Optional[List] = None,
                         init_kwargs: Optional[Dict] = None):
        return CreateInfo(
            self.model_def,
            (self.init_args if self.init_args else []) + (
                init_args if init_args else []),
            (self.init_kwargs if self.init_kwargs else {}).update(
                init_kwargs if init_kwargs else {}),
        )


@dataclasses.dataclass
class ModelInfo:
    create_info: CreateInfo
    group_ids: List[int]
    group_pt: int


@dataclasses.dataclass
class GroupInfo:
    manager: ActorHandle
    queue_size: int
    num_total_requests: int


class DummyRequest:
    """A dummy class to mimic the behavior of starlette.requests.Request."""

    def __init__(self, obj):
        self.obj = obj
        self.scope = {"ts": []}

    async def json(self):
        return self.obj


@ray.remote(num_cpus=1)
class GroupManager:

    def __init__(self, virtual_mesh_shape: Optional[Tuple[int]] = None):
        from alpa.api import init as alpa_init

        if virtual_mesh_shape:
            alpa_init(cluster="ray",
                      num_nodes=virtual_mesh_shape[0],
                      num_devices_per_node=virtual_mesh_shape[1])
        else:
            alpa_init(cluster="ray")

        # Dict[str -> object]
        self.replicas = {}

        # Dict[model_name -> Dict[batch_size -> List[stage_latency]]]
        self.latency_dict = defaultdict(dict)

        # Latency prediction
        self.stage_clock = [0] * np.prod(virtual_mesh_shape)
        self.latency_scale = {}
        self.max_latency_scale = 1.08
        self.freeze_end = 0

        self.logger = build_logger()

        # Constants
        self.fixed_overhead = 0.004

    def create_replica(self, name: str, create_info: CreateInfo):
        assert name not in self.replicas

        model_def, args, kwargs = (create_info.model_def, create_info.init_args,
                                   create_info.init_kwargs)
        args = args or []
        kwargs = kwargs or {}
        self.replicas[name] = model_def(*args, **kwargs)

        if hasattr(self.replicas[name], "get_latency_dict"):
            self.latency_dict[name] = self.replicas[name].get_latency_dict()
        else:
            self.latency_dict[name] = defaultdict(lambda: [0])

    def delete_replica(self, name: str):
        assert name in self.replicas
        del self.replicas[name]

    async def handle_request(self, name: str, request_wrapper: Union[bytes, dict]):
        enter_time = time.time()

        if isinstance(request_wrapper, bytes):
            request_wrapper = pickle.loads(request_wrapper)
            request = build_starlette_request(request_wrapper)
        elif isinstance(request_wrapper, dict):
            request = DummyRequest(request_wrapper)
        else:
            raise ValueError(f"Invalid request type: {request_wrapper}")

        request.scope["ts"].append(("b", enter_time))
        obj = await request.json()

        if "slo" in obj:
            slo = obj["slo"]
            k = self.latency_scale[name]
            # SLO awareness
            stage_latency = self.latency_dict[name][1]

            # Simulate clock
            req_stage_clock = []
            start_time = t = time.time()
            for i in range(len(stage_latency)):
                t = max(self.stage_clock[i], t) + stage_latency[i] * k
                req_stage_clock.append(t)
            ret_time = req_stage_clock[-1]

            # Drop this request if it will exceed deadline
            if ret_time + self.fixed_overhead + 0.001 > obj["submit_time"] + slo:
                return {"rejected": True, "ts": request.scope["ts"]}

            # Accept this request
            for i in range(len(stage_latency)):
                self.stage_clock[i] = req_stage_clock[i]
        else:
            ret_time = None

        try:
            ret = await self.replicas[name].handle_request(request)
        except Exception as e:  # pylint: disable=broad-except
            ret = RelayException(e)

        if ret_time:
            if time.time() + self.fixed_overhead > obj["submit_time"] + slo:
                underestimated = True
            else:
                underestimated = False

            if start_time > self.freeze_end and underestimated:
                actual_runtime = time.time() - start_time
                predicted_runtime = ret_time - start_time
                ratio = actual_runtime / predicted_runtime

                # Adjust the clock to block all requests temporarily
                num_stages = len(stage_latency)
                queue_size = (self.stage_clock[0] - start_time) / (
                    predicted_runtime / num_stages)
                adjust_clock = actual_runtime / num_stages * queue_size / 2
                for i in range(len(stage_latency)):
                    self.stage_clock[i] += adjust_clock
                print(f"adjust clock: {adjust_clock:.2f}, queue size: {queue_size:.2f}, ratio: {ratio:.2f}")

                # Adjust the scale
                if ratio > 1.2:
                    for key in self.latency_scale:
                        self.latency_scale[key] = min(
                            self.max_latency_scale,
                            self.latency_scale[key] + 0.03)
                    print(f"adjust latency scale: {to_str_round(self.latency_scale, 2)}")
                self.freeze_end = self.stage_clock[-1]

        return ret

    async def warmup(self):
        n_iter = 12
        n_warmup = 6
        max_retry = 5

        for name in self.replicas:
            estimated = np.sum(self.latency_dict[name][1])

            retry = 0
            self.latency_scale[name] = math.inf

            while self.latency_scale[name] > self.max_latency_scale and retry < max_retry:
                actual = []
                for i in range(n_iter):
                    start = time.time()
                    request = DummyRequest({"input": "Test."})
                    res = await self.replicas[name].handle_request(request)
                    e2e_latency = time.time() - start
                    actual.append(e2e_latency)

                    #tstamps = {x: (y - start) * 1e3 for x, y in res["ts"]}
                    #print(f"idx: {i}, ts: {to_str_round(tstamps,2)}, "
                    #      f"actual: {e2e_latency*1e3:.2f} ms, "
                    #      f"estimated: {estimated*1e3:.2f} ms")

                self.latency_scale[name] = np.median(actual[n_warmup:]) / estimated
                retry += 1

        for name in self.latency_scale:
            self.latency_scale[name] = np.median(list(self.latency_scale.values()))
        print(f"latency scale: {to_str_round(self.latency_scale, 2)}")

    def shutdown(self):
        from alpa.api import shutdown as alpa_shutdown

        del self.replicas
        alpa_shutdown()


@ray.remote(num_cpus=0)
class Controller:

    def __init__(self,
                 host: str,
                 port: int,
                 root_path: str,
                 ssl_keyfile: Optional[str] = None,
                 ssl_certfile: Optional[Union[str, os.PathLike]] = None):
        self.host = host
        self.port = port
        self.root_path = root_path
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile

        # Controller metadata
        self.manager_lock = defaultdict(asyncio.Lock)

        # Dict[str -> ModelInfo]
        self.model_info = {}
        # Dict[int -> GroupInfo]
        self.group_info = {}

        self.logger = build_logger()

        self.group_manager_class = GroupManager

        # Http server
        self.server = None
        self.setup_complete = asyncio.Event()
        self.http_server_task = asyncio.create_task(self.run_http_server())

    async def create_mesh_group_manager(
            self,
            group_id: int,
            virtual_mesh_shape: Optional[Tuple[int]] = None,
            num_gpus: int = 0):
        assert group_id not in self.group_info, (
            f"Mesh group {group_id} is already launched")
        self.logger.info(f"Create mesh group manager {group_id} with "
                         f"shape={virtual_mesh_shape}")
        manager = (self.group_manager_class.options(
            name=f"mesh_group_manager_{group_id}",
            num_gpus=num_gpus).remote(virtual_mesh_shape))
        self.group_info[group_id] = GroupInfo(
            manager=manager, queue_size=0, num_total_requests=0)

    async def register_model(self,
                             name: str,
                             model_def: Callable,
                             init_args: Optional[List] = None,
                             init_kwargs: Optional[Dict] = None,
                             override: bool = False):
        async with self.manager_lock[name]:
            if name in self.model_info:
                if override:
                    for group_id in self.model_info[name].group_ids:
                        await self.group_info[group_id
                            ].manager.delete_replica.remote(name)
                else:
                    raise ValueError(f"Model {name} is already registered")

            self.model_info[name] = ModelInfo(
                CreateInfo(model_def, init_args, init_kwargs), [], 0)

    async def create_replica(self,
                             name: str,
                             group_id: int,
                             append_init_args: Optional[List] = None,
                             append_init_kwargs: Optional[Dict] = None):
        async with self.manager_lock[name]:
            assert group_id in self.group_info, (
                f"Group {group_id} does not exist")
            model_info = self.model_info[name]
            manager = self.group_info[group_id].manager
            assert group_id not in model_info.group_ids, (
                f"Model {name} is already created on group {group_id}")
            create_info = model_info.create_info.append_init_args(
                append_init_args, append_init_kwargs)

            self.logger.info(f"Create replica of {name} on group {group_id}")
            model_info.group_ids.append(group_id)
        await manager.create_replica.remote(name, create_info)

    def select_group_id(self, group_ids):
        min_id = -1
        min_size = math.inf
        for group_id in group_ids:
            if self.group_info[group_id].queue_size < min_size:
                min_size = self.group_info[group_id].queue_size
                min_id = group_id
        assert min_id != -1
        return min_id

    async def handle_request(self, request):
        ts = [("a", time.time())]
        name = request["model"]

        assert name in self.model_info, (
            f"Model '{name}' is not registered.")
        model_info = self.model_info[name]
        #assert model_info.group_ids, (
        #    f"No replica of model '{name}' is created.")

        if not model_info.group_ids:
            return {"rejected": True}
        else:
            # Dispatch
            group_id = self.select_group_id(model_info.group_ids)
            manager = self.group_info[group_id].manager

            self.group_info[group_id].queue_size += 1
            response = await manager.handle_request.remote(name, request)
            self.group_info[group_id].queue_size -= 1
            self.group_info[group_id].num_total_requests += 1

            response["ts"] = ts + response["ts"]

            return response

    async def handle_asgi(self, scope, receive, send):
        scope["ts"] = [("a", time.time())]
        assert scope["type"] == "http"

        # Receive request
        http_body_bytes = await receive_http_body(scope, receive, send)
        request_wrapper = HTTPRequestWrapper(scope, http_body_bytes)
        request_wrapper_bytes = pickle.dumps(request_wrapper)

        query_params = QueryParams(scope["query_string"])

        # Route
        try:
            if "model" in query_params:
                name = query_params["model"]
            else:
                request = build_starlette_request(request_wrapper)
                obj = await request.json()

                assert "model" in obj, "Model name is not specified in the request."
                name = obj["model"]

            assert name in self.model_info, (
                f"Model '{name}' is not registered.")
            model_info = self.model_info[name]
            #assert model_info.group_ids, (
            #    f"No replica of model '{name}' is created.")

            if not model_info.group_ids:
                status_code = 200
                response = {"rejected": True}
            else:
                # Dispatch
                group_id = self.select_group_id(model_info.group_ids)
                manager = self.group_info[group_id].manager

                self.group_info[group_id].queue_size += 1
                response = await manager.handle_request.remote(
                    name, request_wrapper_bytes)
                self.group_info[group_id].queue_size -= 1
                self.group_info[group_id].num_total_requests += 1

                if isinstance(response, RelayException):
                    response = make_error_response(response)
                    status_code = 400
                else:
                    status_code = 200
        except Exception as e:  # pylint: disable=broad-except
            response = make_error_response(e)
            status_code = 400

        await Response(response,
                       status_code=status_code).send(scope, receive, send)

    def get_info(self):
        return {
            "host": self.host,
            "port": self.port,
            "root_path": self.root_path,
        }

    async def warmup(self):
        # Warm up each single model replica in each group
        tasks = []
        for g in self.group_info.values():
            tasks.append(g.manager.warmup.remote())
        ray.get(tasks)

        # Warm up the whole path from the controller to groups
        for name, info in self.model_info.items():
            request = {"model": name, "input": "Test"}
            objs = []
            for i in range(len(info.group_ids)):
                objs.append(self.handle_request(request))
            await asyncio.gather(*objs)

    ##### HTTP related functions #####
    async def ready(self):
        """Returns when HTTP proxy is ready to serve traffic.
        Or throw exception when it is not able to serve traffic.
        """
        done_set, _ = await asyncio.wait(
            [
                # Either the HTTP setup has completed.
                # The event is set inside self.run.
                self.setup_complete.wait(),
                # Or self.run errored.
                self.http_server_task,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Return None, or re-throw the exception from self.running_task.
        return await done_set.pop()

    async def run_http_server(self):
        sock = socket.socket()
        if SOCKET_REUSE_PORT_ENABLED:
            set_socket_reuse_port(sock)

        try:
            sock.bind((self.host, self.port))
        except OSError:
            # The OS failed to bind a socket to the given host and port.
            raise ValueError(
                f"Failed to bind HTTP proxy to '{self.host}:{self.port}'."
                f"Please make sure your http-host and http-port are "
                f"specified correctly.")

        # Note(simon): we have to use lower level uvicorn Config and Server
        # class because we want to run the server as a coroutine. The only
        # alternative is to call uvicorn.run which is blocking.
        app = ASGIHandler(self)
        #app = CORSMiddleware(
        #    app,
        #    allow_origins=["*"],
        #    allow_methods=["*"],
        #    allow_headers=["*"],
        #)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            root_path=self.root_path,
            lifespan="off",
            access_log=False,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
        )
        self.server = uvicorn.Server(config=config)

        # TODO(edoakes): we need to override install_signal_handlers here
        # because the existing implementation fails if it isn't running in
        # the main thread and uvicorn doesn't expose a way to configure it.
        self.server.install_signal_handlers = lambda: None

        self.setup_complete.set()
        await self.server.serve(sockets=[sock])

    async def shutdown(self):
        if self.server is not None:
            self.server.should_exit = True
        tasks = []
        for g in self.group_info.values():
            tasks.append(g.manager.shutdown.remote())
        await asyncio.sleep(0.5)
        await asyncio.gather(*tasks)


def run_controller(host,
                   port=None,
                   root_path="/",
                   name=CONTROLLER_NAME,
                   ssl_keyfile: Optional[str] = None,
                   ssl_certfile: Optional[Union[str, os.PathLike]] = None):
    """Launch a controller"""
    controller = Controller.options(
        name=name,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )).remote(
            host=host,
            port=port or new_port(),
            root_path=root_path,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    ray.get(controller.ready.remote())

    add_sync_method(controller,
                    ["create_replica", "create_mesh_group_manager", "warmup"])
    return controller
