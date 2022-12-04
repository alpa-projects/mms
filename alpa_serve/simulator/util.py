"""Common utilities"""
import asyncio
from functools import partial
import threading

import numpy as np


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


# Workload generation utils


class MMPPSampler:
    """Sample a sequence of requests from a Markov Modulated Poisson Process."""
    def __init__(self, Q, lambda_):
        """Initialize a MMPP sampler.

        Args:
            Q (np.ndarray): Transition matrix of the Markov chain.
            lambda_ (np.ndarray): Lambdas of the Poisson process of each state.
        """
        self.Q = Q
        self.lambda_ = lambda_
        self.m = Q.shape[0]
        assert Q.shape == (self.m, self.m)
        assert lambda_.shape == (self.m,)
        self.Pi = np.identity(self.m) - np.diag(1 / np.diag(self.Q)) @ self.Q

    def sample(self, num_requests, initial_state=0):
        """Generate samples using the Markov-modulated Poisson process.

        Args:
            num_requests (int): Number of requests to generate.
            initial_state (int): Initial state of the Markov chain.

        Returns:
            tau: Arrival times of the requests.
            y: The duration of each state.
            y: The state sequence.
            ys: States of the individual requests.
        """
        assert 0 <= initial_state < self.m
        ys = [initial_state]
        x = [0]
        y = [initial_state]
        tau = [0]
        while True:
            state = y[-1]
            y.append(np.random.choice(self.m, p=self.Pi[state]))
            t = x[-1]
            x.append(t + np.random.exponential(-1 / self.Q[state, state]))
            while True:
                t = t + np.random.exponential(1 / self.lambda_[state])
                if t > x[-1]:
                    break
                tau.append(t)
                ys.append(state)
                if len(tau) == num_requests + 1:
                    return tau, (x, y, ys)

    def expected_request_rate(self):
        """Compute the expected request rate."""
        return self.lambda_ @ self.Pi

    @classmethod
    def unifrom_mmpp(cls, expected_state_durations,
                     expected_state_request_rates):
        """Special case of MMPP where the transition matrix from one state to
        another is uniform.

        Args:
            num_requests (int): Number of requests to generate.
            expected_state_durations (np.ndarray): Expected durations of each
                state.
            expected_state_request_rates (np.ndarray): Expected request rates of
                each state.
            initial_state (int): Initial state of the Markov chain.
        """
        m = len(expected_state_durations)
        assert len(expected_state_request_rates) == m
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    Q[i, j] = -1 / expected_state_durations[i]
                else:
                    Q[i, j] = 1 / expected_state_durations[i] / (m - 1)
        lambda_ = np.array(expected_state_request_rates)
        return cls(Q, lambda_)
