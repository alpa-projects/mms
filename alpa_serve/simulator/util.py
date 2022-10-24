import asyncio
from functools import partial


def install_remote_methods(x):
    for key in dir(x):
        value = getattr(x, key)
        if callable(value) and key[0] != "_":
            if asyncio.iscoroutinefunction(value):
                raise NotImplementedError()
            else:
                new_value = partial(value)
            setattr(new_value, "remote", new_value)
            setattr(x, key, new_value)

