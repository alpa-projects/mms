"""Common utilities."""
from collections import namedtuple
import functools
import logging
import math
from typing import Sequence, Any

import ray
import numpy as np

# global switch for batching
# enable_batching = True
batchsize_config = [1, 2, 4, 8, 16]

# A general serving case.
# We can simulate or run such a case.
ServingCase = namedtuple("ServingCase",
    ("register_models", "generate_workload", "placement_policy"))


GB = 1 << 30
eps = 1e-6
inf = 1e100


def build_logger(name="alpa_serve"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def add_sync_method(actor, method_names):
    """Add a actor.sync method to wait for all calls to methods
    listed in method_names."""
    calls = []

    def sync():
        ray.get(calls)
        calls.clear()

    setattr(actor, "sync", sync)

    for name in method_names:
        attr = getattr(actor, name)
        old_remote = attr.remote
        setattr(attr, "remote", functools.partial(wrapped_remote_call, old_remote, calls))


def wrapped_remote_call(old_remote, calls, *args, **kwargs):
    ret = old_remote(*args, *kwargs)
    calls.append(ret)
    return ret


def write_tsv(heads: Sequence[str],
              values: Sequence[Any],
              filename: str,
              print_line: bool = True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    values = [str(x) for x in values]

    with open(filename, "a", encoding="utf-8") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def get_factors(n: int):
    step = 2 if n % 2 else 1
    ret = list(
        set(
            functools.reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(math.sqrt(n)) + 1, step) if n % i == 0),
            )
        )
    )
    ret.sort()
    return ret


def to_str_round(x: Any, decimal: int = 6):
    """Print a python object but round all floating point numbers."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp_str = ", ".join([to_str_round(y, decimal=decimal) for y in x])
        return "[" + tmp_str + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v, decimal=decimal) for k, v in x.items()})
    if isinstance(x, (int, np.int32, np.int64)):
        return str(x)
    if isinstance(x, (float, np.float32, np.float64)):
        format_str = f"%.{decimal}f"
        return format_str % x
    if x is None:
        return str(x)
    raise ValueError("Invalid value: " + str(x))


def is_valid_size(n: int, i: int):
    if i <= n % 8 or (n - i) % 8 == 0 or n % 8 == 0:
        return True
    else:
        return False

# partition n into k parts that summed to n
# each part could only be 2^k
def get_partitions(n: int, k: int):
    if k == 1:
        return [[n]]

    ret = []
    for i in range(1, n):
        if not is_valid_size(n, i): continue
        pre_partitions = get_partitions(n - i, k - 1)
        ret += [partition + [i] for partition in pre_partitions]
    return ret


#def get_partitions(n: int, k: int, lb: int = 1):
#    if k == 1:
#        if n >= lb:
#            return [[n]]
#        else:
#            return []
#
#    ret = []
#    for i in range(lb, n):
#        if not is_valid_size(n, i): continue
#        pre_partitions = get_partitions(n - i, k - 1, i)
#        ret += [partition + [i] for partition in pre_partitions]
#    return ret


def get2tok(n: int):
    assert n > 0
    ret = [1]
    while True:
        if ret[-1] * 2 <= n:
            ret.append(ret[-1] * 2)
        else:
            break
    return ret


def decompose2tok(n: int):
    ret = []
    i = 1
    while n > 0:
        if n % 2 == 1:
            ret.append(i)
        i *= 2
        n = n // 2
    return ret


if __name__ == "__main__":
    print(get_partitions(64, 6))
    print(len(get_partitions(64, 6)))
    print(get2tok(34))
    print(decompose2tok(13))
