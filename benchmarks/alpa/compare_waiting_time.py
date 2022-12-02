from alpa_serve.simulator.workload import GammaProcess


def kingman_formula(arrival_rate, arrival_CV, service_rate):
    p = arrival_rate / service_rate
    assert 0 <= p <= 1
    return p / (1 - p) * (arrival_CV ** 2) / 2 * (1 / service_rate)


def waiting_time(workload, service_time):
    return kingman_formula(workload.rate, workload.cv, 1 / service_time) + service_time


def pipeline_waiting_time(workload, stage_service_time):
    return kingman_formula(workload.rate, workload.cv, 1 / max(stage_service_time)) + sum(stage_service_time)


if __name__ == "__main__":
    r_a = GammaProcess(3, 2).generate_workload("a", start=0, duration=1000, seed=10)
    r_b = GammaProcess(3, 2).generate_workload("b", start=0, duration=1000, seed=11)
    r_c = GammaProcess(3, 2).generate_workload("c", start=0, duration=1000, seed=12)
    r_d = GammaProcess(3, 2).generate_workload("d", start=0, duration=1000, seed=13)

    # replication 1x
    w1 = waiting_time(r_a + r_b, 0.1)
    w2 = waiting_time(r_c + r_d, 0.1)
    print(f"w1: {w1: .3f}, w2: {w2:.3f}")

    # replication 2x
    w1 = waiting_time(r_a[::2] + r_b[::2] + r_c[::2] + r_d[::2], 0.1)
    w2 = waiting_time(r_a[1::2] + r_b[1::2] + r_c[1::2] + r_d[1::2], 0.1)
    print(f"w1: {w1: .3f}, w2: {w2:.3f}")
    r = r_a[::2] + r_b[::2] + r_c[::2] + r_d[::2]
    print(f"rate: {r.rate: .3f}, cv: {r.cv: .3f}")

    # pipeline 2x
    w1 = pipeline_waiting_time(r_a + r_b + r_c + r_d, [0.052, 0.050])
    print(f"w1: {w1: .3f}")
    r = r_a + r_b + r_c + r_d
    print(f"rate: {r.rate: .3f}, cv: {r.cv: .3f}")
