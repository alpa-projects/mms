"""A discrete event simulator that supports asyncio"""
import asyncio
from collections import defaultdict
from enum import Enum, auto
from functools import partial
import heapq
import queue
import time
from typing import Callable, List, Dict, Union, Sequence


class CoroutineStatus(Enum):
    INIT = auto()
    PAUSE = auto()
    FINISH = auto()


class TimedCoroutine:
    """A coroutine that will be woken up at specific time."""
    def __init__(self,
                 wake_up_time: float,
                 func: Callable):
        self.wake_up_time = wake_up_time
        self.func = func
        self.status = CoroutineStatus.INIT

        self.atask = None
        self.afuture = None
        self.resume_event = None
        self.resume_future = None
        self.resume_future_value = None
        self.waiter = None

        self.ret_value = None

    def __lt__(self, other):
        return self.wake_up_time < other.wake_up_time

    def __str__(self):
        if hasattr(self.func, "__name__"):
            name = self.func.__name__
        elif hasattr(self.func, "func"):
            name = self.func.func.__name__
        else:
            name = ""
        return f"TimedCoroutine(wake_up_time={self.wake_up_time}, func={name})"


class Stream:
    """A stream resource."""
    def __init__(self):
        self.clock = 0


class PriorityQueue:
    def __init__(self):
        self.store = []

    def put(self, value):
        heapq.heappush(self.store, value)

    def get(self):
        return heapq.heappop(self.store)

    def __len__(self):
        return len(self.store)

    def __bool__(self):
        return True if self.store else False


class EventLoop:
    """The main event loop"""
    def __init__(self):
        self.queue = PriorityQueue()
        self.clock_ = 0
        self.cur_tc = None  # The current TimedCoroutine
        self.pause_event = asyncio.Event()

        self.streams = defaultdict(Stream)

        self.main_loop = asyncio.create_task(self.run())

    async def run(self):
        while self.queue:
            tc = self.queue.get()
            self.cur_tc = tc

            self.clock_ = tc.wake_up_time

            self.pause_event.clear()

            if tc.status == CoroutineStatus.INIT:
                coroutine = tc.func()
                atask = asyncio.create_task(coroutine)
                tc.atask = atask
            elif tc.status == CoroutineStatus.PAUSE:
                atask = tc.atask
                if tc.resume_event:
                    tc.resume_event.set()
                    tc.resume_event = None
                elif tc.resume_future:
                    tc.resume_future.set_result(tc.resume_future_value)
                    tc.resume_future = None
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            done, pending = await asyncio.wait([atask, self.pause_event.wait()],
                return_when=asyncio.FIRST_COMPLETED)

            if atask.done():
                if atask.exception():
                    exception = atask.exception()
                    print(f"Exception: {exception}")

                tc.status = CoroutineStatus.FINISH
                tc.ret_value = await list(done)[0]

                if tc.afuture:
                    tc.afuture.set_result(tc.ret_value)

                if tc.waiter:
                    w = tc.waiter
                    w.wake_up_time = self.clock_
                    w.resume_future_value = tc.ret_value
                    self.queue.put(w)

    def put_coroutine(self, tstamp: float, func: Callable, args: List, kwargs: Dict):
        if self.cur_tc:
            tc = self.cur_tc

            new_tc = TimedCoroutine(tstamp, partial(func, *args, **kwargs))
            new_tc.waiter = tc
            self.queue.put(new_tc)

            self.pause_event.set()
            tc.status = CoroutineStatus.PAUSE
            tc.resume_future = asyncio.get_running_loop().create_future()
            return tc.resume_future
        else:
            new_tc = TimedCoroutine(tstamp, partial(func, *args, **kwargs))
            new_tc.afuture = asyncio.get_running_loop().create_future()
            self.queue.put(new_tc)
            return new_tc.afuture

    def sleep(self, duration: float):
        assert duration >= 0
        self.pause_event.set()

        tc = self.cur_tc
        tc.wake_up_time = self.clock_ + duration
        tc.status = CoroutineStatus.PAUSE
        tc.resume_event = asyncio.Event()
        self.queue.put(tc)
        return tc.resume_event.wait()

    def wait_stream(self, name: Union[str, int], duration: float):
        assert duration >= 0

        stream = self.streams[name]
        stream.clock = max(stream.clock, self.clock_) + duration

        return self.sleep(stream.clock - self.clock_)

    def wait_multi_stream(self, names: Sequence[Union[str, int]],
                          durations: Sequence[float]):
        assert all(d >= 0 for d in durations)
        assert len(names) == len(durations)

        max_clock = -1
        for i in range(len(names)):
            stream = self.streams[names[i]]
            stream.clock = max(stream.clock, self.clock_) + durations[i]
            max_clock = max(max_clock, stream.clock)

        return self.sleep(max_clock - self.clock_)

    def clock(self):
        return self.clock_


loop = None

def run_event_loop(coroutine):
    """Run and simulate an event loop"""
    async def main():
        global loop
        loop = EventLoop()
        ret = await coroutine
        await loop.main_loop
        return ret

    return asyncio.run(main())


clock = lambda: loop.clock()
sleep = lambda *args: loop.sleep(*args)
wait_stream = lambda *args: loop.wait_stream(*args)
wait_multi_stream = lambda *args: loop.wait_multi_stream(*args)
main_loop = lambda: loop.main_loop


def timed_coroutine(func):
    """Convert a coroutine function to a timed coroutine function for simulation."""
    assert asyncio.iscoroutinefunction(func)

    def ret_func(*args, **kwargs):
        if "tstamp" in kwargs:
            tstamp = kwargs.pop("tstamp")
        elif "delay" in kwargs:
            tstamp = kwargs.pop("delay") + loop.clock()
        else:
            tstamp = loop.clock()
        return loop.put_coroutine(tstamp, func, args, kwargs)

    return ret_func


@timed_coroutine
async def call_model():
    print("call_model", clock(), flush=True)
    await wait_stream("gpu", 10)
    return "answer"


@timed_coroutine
async def call_controller():
    print("call_controller begin", clock(), flush=True)
    x = await call_model(delay=5)
    print("call_controller end", clock(), flush=True)
    return x


async def test_main():
    call_controller(tstamp=1)
    x = call_controller(tstamp=1)

    assert (await x) == "answer"


if __name__ == "__main__":
    run_event_loop(test_main())
