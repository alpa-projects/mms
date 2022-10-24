"""A discrete event simulator that supports asyncio"""
import asyncio
from collections import defaultdict
from enum import Enum, auto
import dataclasses
from functools import partial
import queue
from typing import Callable, Any, List, Dict, Optional, Union

eps = 1e-15


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

        self.start = None
        self.end = None
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


class EventLoop:
    """The main event loop"""
    def __init__(self):
        self.queue = asyncio.PriorityQueue()
        self.clock_ = 0
        self.cur_tc = None
        self.pause_event = asyncio.Event()

        self.streams = defaultdict(Stream)

        self.main_task = asyncio.create_task(self.run())

    async def run(self):
        while not self.queue.empty():
            tc = await self.queue.get()
            self.cur_tc = tc

            self.clock_ = tc.wake_up_time

            self.pause_event.clear()

            if tc.status == CoroutineStatus.INIT:
                coroutine = tc.func()
                atask = asyncio.create_task(coroutine)
                tc.atask = atask
                tc.start = self.clock_
            elif tc.status == CoroutineStatus.PAUSE:
                atask = tc.atask
                if tc.resume_event:
                    tc.resume_event.set()
                elif tc.resume_future:
                    tc.resume_future.set_result(tc.resume_future_value)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            done, pending = await asyncio.wait([atask, self.pause_event.wait()],
                return_when=asyncio.FIRST_COMPLETED)

            if tc.atask.done():
                tc.status = CoroutineStatus.FINISH
                tc.ret_value = await list(done)[0]
                if tc.afuture:
                    tc.afuture.set_result(tc.ret_value)
                tc.end = self.clock_

                if tc.waiter:
                    w = tc.waiter
                    w.wake_up_time = self.clock_
                    w.resume_future_value = tc.ret_value
                    self.queue.put_nowait(w)

    def put_coroutine(self, tstamp: float, func: Callable, args: List, kwargs: Dict):
        if self.cur_tc:
            tc = self.cur_tc

            new_tc = TimedCoroutine(tstamp, partial(func, *args, **kwargs))
            new_tc.waiter = tc
            self.queue.put_nowait(new_tc)

            self.pause_event.set()
            tc.status = CoroutineStatus.PAUSE
            tc.resume_future = asyncio.get_running_loop().create_future()
            return tc.resume_future
        else:
            new_tc = TimedCoroutine(tstamp, partial(func, *args, **kwargs))
            new_tc.afuture = asyncio.get_running_loop().create_future()
            self.queue.put_nowait(new_tc)
            return new_tc.afuture

    def sleep(self, duration: float):
        assert duration >= 0
        self.pause_event.set()

        tc = self.cur_tc
        tc.wake_up_time = self.clock_ + duration
        tc.status = CoroutineStatus.PAUSE
        tc.resume_event = asyncio.Event()
        self.queue.put_nowait(tc)
        return tc.resume_event.wait()

    def wait_stream(self, name: Union[str, int], duration: float):
        assert duration >= 0

        stream = self.streams[name]
        stream.clock = max(stream.clock, self.clock_) + duration

        return self.sleep(stream.clock - self.clock_)

    def clock(self):
        return self.clock_



loop = None
sleep = clock = wait_stream = None


def init_event_loop():
    global loop, sleep, clock, wait_stream
    loop = EventLoop()
    clock = loop.clock
    sleep = loop.sleep
    wait_stream = loop.wait_stream


def timed_coroutine(func):
    def ret_func(*args, **kwargs):
        if "tstamp" in kwargs:
            tstamp = kwargs.pop("tstamp")
        elif "delay" in kwargs:
            tstamp = kwargs.pop("delay") + loop.clock()
        return loop.put_coroutine(tstamp, func, args, kwargs)

    return ret_func


@timed_coroutine
async def low_event():
    print("low event", clock(), flush=True)
    await wait_stream("gpu", 10)
    return "low"


@timed_coroutine
async def high_event():
    print("high level begin", clock(), flush=True)
    x = await low_event(delay=5)
    assert x == "low"
    print("high level end", clock(), flush=True)

    return "high"


async def test_main():
    init_event_loop()

    high_event(tstamp=1)
    x = high_event(tstamp=1)

    assert (await x == "high")


if __name__ == "__main__":
    asyncio.run(test_main())
