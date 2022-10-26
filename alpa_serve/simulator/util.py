"""Common utilities"""
import asyncio
from functools import partial
import threading


def install_remote_methods(x):
    """Map obj.func.remote to obj.func, so we can create fake Ray APIs."""
    for key in dir(x):
        value = getattr(x, key)
        if callable(value) and key[0] != "_":
            new_value = partial(value)
            setattr(new_value, "remote", new_value)
            setattr(x, key, new_value)


def async_to_sync(async_def):
    """Convert a coroutine function to a normal function."""
    assert asyncio.iscoroutinefunction(async_def)

    def ret_func(*args, **kwargs):
        corountine = async_def(*args, **kwargs)
        return run_coroutine(corountine)

    return ret_func


def run_coroutine(corountine):
    """Run an asynchronous corountine synchronously."""
    ret = []

    def target():
        ret.append(asyncio.run(corountine))

    # Start a new thread to allow nested asyncio loops
    t = threading.Thread(target=target)
    t.start()
    t.join()

    return ret[0]
