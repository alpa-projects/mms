import json
import unittest
import subprocess

import numpy as np

from alpasim.cluster import Cluster, Mesh, load_meshexecutors, save_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.workload import generate_workload, PossoinWorkLoad
from alpasim.utils import compute_statistics_from_cluster_trace, compute_statistics_from_simulation, \
                          dump_chrome_tracing_from_simulation, dump_chrome_tracing_from_cluster_trace

VERBOSE=True

class Test(unittest.TestCase):
    def test_workload_save_load(self):
        workload = generate_workload(2, 10, [0.5, 0.5], 20, [0.25, 0.25], "./workload/Test workload")
        workload.plot(0.5, "before save")
        del workload
        workload = PossoinWorkLoad.load("./workload/Test workload")
        workload.plot(0.5, "after load")
        ret = subprocess.run(["diff", "before save.png", "after load.png"])
        # clean the temp files before check
        subprocess.run(["rm", "./workload/Test workload", "before save.png", "after load.png"])
        self.assertEqual(ret.returncode, 0)
    
    def test_placement_save_load(self):
        placement_filename = "./placements/placement_test.json"
        cluster = Cluster(1, 4, 16)
        meshexecutorsmap = load_meshexecutors(placement_filename, cluster)
        meshexecutors = []
        for execs in meshexecutorsmap.values():
            meshexecutors.extend(execs)
        save_meshexecutors(meshexecutors, "placement_check.json")
        ret = subprocess.run(["diff", placement_filename, "placement_check.json"])
        # clean the temp files before check
        subprocess.run(["rm", "placement_check.json"])
        self.assertEqual(ret.returncode, 0)
    
    def test_mesh(self):
        cluster = Cluster(1, 4, 16)
        mesh = Mesh(cluster, [0], [[0, 1, 2]])
        self.assertEqual(mesh.shape, (1, 3))
        gpus = mesh.get_all_gpus()
        self.assertEqual(len(gpus), 3)
        for gpu in gpus:
            self.assertEqual(gpu.next_idle_time, 0.0)
            self.assertEqual(gpu.next_event_time, float('inf'))
    
    def check_simulation_error(self, cluster_trace_filename, 
                                     simulated_tasks, 
                                     assertion=True):
        s_cluster, e_cluster = {}, {}
        with open(cluster_trace_filename, "r") as f:
            traces = json.load(f)
            for _, trace in traces.items():
                for stage_exec_info in trace["stage_exec_info"]:
                    for rq_id, (s, e, _, _) in zip(trace["rq_id"], stage_exec_info):
                        if rq_id not in s_cluster:
                            s_cluster[rq_id] = [s]
                            e_cluster[rq_id] = [e]
                        else:
                            s_cluster[rq_id].append(s)
                            e_cluster[rq_id].append(e)

        s_sim, e_sim = {}, {}
        for i, t in enumerate(simulated_tasks):
            s_sim[i], e_sim[i] = [], []
            for stage_exec_info in t.stage_execution_info:
                s, e, _, _ = stage_exec_info[0]
                s_sim[i].append(s)
                e_sim[i].append(e)

        self.assertEqual(s_sim.keys(), s_cluster.keys())
        num_stages = len(s_sim[0])
        s_error = [[] for _ in range(num_stages)]
        e_error = [[] for _ in range(num_stages)]

        for i in s_sim.keys():
            for stage_num, (s1, s2) in enumerate(zip(s_sim[i], s_cluster[i])):
                s_error[stage_num].append(abs(s1 - s2))
                if VERBOSE:
                    if (abs(s1 - s2) > 0.01):
                        print(f"Request {i}, stage {stage_num}, simu start: {s1:.5f}, cluster start: {s2:.5f}")
                if assertion:
                    self.assertLessEqual(abs(s1 - s2), 0.01)
            for stage_num, (e1, e2) in enumerate(zip(e_sim[i], e_cluster[i])):
                if VERBOSE:
                    if (abs(e1 - e2) > 0.01):
                        print(f"Request {i}, stage {stage_num}, simu end: {e1:.5f}, cluster end: {e2:.5f}")
                e_error[stage_num].append(abs(e1 - e2))
                if assertion:
                    self.assertLessEqual(abs(e1 - e2), 0.01)

        if VERBOSE:
            print("------------------------------")
            for num_stage, error in enumerate(s_error):
                print(f"Stage {num_stage} start mean error: {np.mean(error):.5f}s, max error: {np.max(error):.5f}s")
            print("------------------------------")
            for num_stage, error in enumerate(e_error):
                print(f"Stage {num_stage} end mean error: {np.mean(error):.5f}s, max error: {np.max(error):.5f}s")


    def test_simulation(self, workload_name, 
                              placement_filename, 
                              model_id_to_service_name, 
                              cluster_trace_filename, 
                              simu_chrome_trace_filename, 
                              cluster_chrome_trace_filename, 
                              assertion=True):
        """ Test simulation accuracy. 

            @param workload_name: name of the workload
            @param placement_filename: placement filename
            @param cluster_trace_filename: the ground truth trace to compared with
            @param simu_chrome_trace_filename: filename of the output chrome trace for simulation
            @param cluster_chrome_trace_filename: filename of the output chrome trace for cluster trace

        """
        workload_filename = f"./workload/{workload_name}"
        workload = PossoinWorkLoad.load(workload_filename)
        cluster = Cluster(1, 2, 16)
        meshexecutors = load_meshexecutors(placement_filename, cluster)
        scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
        simulator = Simulator(scheduler, cluster)
        simulator.start()
        dump_chrome_tracing_from_simulation(scheduler.completed_tasks, simu_chrome_trace_filename)
        dump_chrome_tracing_from_cluster_trace(cluster_trace_filename, cluster_chrome_trace_filename)
        compute_statistics_from_simulation(scheduler.completed_tasks)
        compute_statistics_from_cluster_trace(cluster_trace_filename)
        self.check_simulation_error(cluster_trace_filename, scheduler.completed_tasks, assertion)
 
    def test_2dot6B_Bert_baseline_trace(self):
        workload_name = "test_workload_8to2_6.667Hz_20s"
        placement_filename = "./placements/placement_baseline.json"
        simu_chrome_trace_filename = "chrome_trace/baseline_simu.json"
        cluster_trace_filename = f"./cluster_traces/{workload_name}_baseline_trace.json"
        cluster_chrome_trace_filename = "chrome_trace/baseline_cluster.json"
        model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
        print("\n========================")
        print("Test baseline trace:")
        self.test_simulation(workload_name, placement_filename, model_id_to_service_name, cluster_trace_filename, simu_chrome_trace_filename, cluster_chrome_trace_filename)
  
    def test_2dot6B_Bert_inter_op_trace(self):
        workload_name = "test_workload_8to2_6.667Hz_20s"
        placement_filename = "./placements/placement_interop.json"
        simu_chrome_trace_filename = "chrome_trace/interop_simu.json"
        cluster_trace_filename = f"./cluster_traces/{workload_name}_interop_trace.json"
        cluster_chrome_trace_filename = "chrome_trace/interop_cluster.json"
        model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
        print("\n========================")
        print("Test interop trace:")
        self.test_simulation(workload_name, placement_filename, model_id_to_service_name, cluster_trace_filename, simu_chrome_trace_filename, cluster_chrome_trace_filename)
   
    def test_2dot6B_Bert_intra_op_trace(self):
        workload_name = "test_workload_8to2_6.667Hz_20s"
        placement_filename = "./placements/placement_intraop.json"
        simu_chrome_trace_filename = "chrome_trace/intraop_simu.json"
        cluster_trace_filename = f"./cluster_traces/{workload_name}_intraop_trace.json"
        cluster_chrome_trace_filename = "chrome_trace/intraop_cluster.json"
        model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
        print("\n========================")
        print("Test intraop trace:")
        self.test_simulation(workload_name, placement_filename, model_id_to_service_name, cluster_trace_filename, simu_chrome_trace_filename, cluster_chrome_trace_filename)
 
    def test_125M_Bert_inter_op_trace(self):
        workload_name = "test_workload_8to2_30Hz_60s"
        placement_filename = "./placements/placement_125M_interop.json"
        simu_chrome_trace_filename = "chrome_trace/125M_interop_simu.json"
        cluster_trace_filename = f"./cluster_traces/{workload_name}_interop_trace.json"
        cluster_chrome_trace_filename = "chrome_trace/125M_interop_cluster.json"
        model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
        print("\n========================")
        print("Test 125M interop trace:")
        self.test_simulation(workload_name, placement_filename, model_id_to_service_name, cluster_trace_filename, simu_chrome_trace_filename, cluster_chrome_trace_filename, False)

    def test_125M_Bert_intra_op_trace(self):
        workload_name = "test_workload_8to2_30Hz_60s"
        placement_filename = "./placements/placement_125M_intraop.json"
        simu_chrome_trace_filename = "chrome_trace/125M_intraop_simu.json"
        cluster_trace_filename = f"./cluster_traces/{workload_name}_intraop_trace.json"
        cluster_chrome_trace_filename = "chrome_trace/125M_intraop_cluster.json"
        model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
        print("\n========================")
        print("Test 125M intraop trace:")
        self.test_simulation(workload_name, placement_filename, model_id_to_service_name, cluster_trace_filename, simu_chrome_trace_filename, cluster_chrome_trace_filename, False)
    
    def test_125M_Bert_strong_baseline(self):
        workload_name = "test_workload_8to2_50Hz_60s"
        placement_filename = "./placements/placement_125M_strong_baseline.json"
        simu_chrome_trace_filename = "chrome_trace/125M_strong_baseline_simu.json"
        model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
        print("\n========================")
        print("Test 125M baseline trace:")
        workload_filename = f"./workload/{workload_name}"
        workload = PossoinWorkLoad.load(workload_filename)
        cluster = Cluster(1, 2, 16)
        meshexecutors = load_meshexecutors(placement_filename, cluster)
        scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
        simulator = Simulator(scheduler, cluster)
        simulator.start()
        dump_chrome_tracing_from_simulation(scheduler.completed_tasks, simu_chrome_trace_filename)
        compute_statistics_from_simulation(scheduler.completed_tasks)

  
def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test('test_workload_save_load'))
    suite.addTest(Test('test_placement_save_load'))
    suite.addTest(Test('test_mesh'))
    suite.addTest(Test('test_2dot6B_Bert_baseline_trace'))
    suite.addTest(Test('test_2dot6B_Bert_inter_op_trace'))
    suite.addTest(Test('test_2dot6B_Bert_intra_op_trace'))
    
    # suite.addTest(Test('test_125M_Bert_strong_baseline'))
    suite.addTest(Test('test_125M_Bert_inter_op_trace'))
    suite.addTest(Test('test_125M_Bert_intra_op_trace'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())