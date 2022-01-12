from GeneralTestTrain import test_kernel_vehicle, load_conf

from SandboxSafety.Simulator.ForestSim import ForestSim
from SandboxSafety.KernelGenerator import construct_obs_kernel, construct_kernel_sides
from SandboxSafety.NavAgents.SimplePlanners import RandomPlanner, PurePursuit 
from SandboxSafety.SupervisorySystem import Supervisor, ForestKernel

import yaml
from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"
kernel_name = f"kernel_sap_{run_n}"



def rando_test():
    conf = load_conf("forest_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = RandomPlanner()
    kernel = ForestKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 20, add_obs=True, wait=False)
    # test_kernel_vehicle(env, safety_planner, False, 100, wait=False)

def pp_test():
    conf = load_conf("forest_kernel")

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    kernel = ForestKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, True, 10, add_obs=True, wait=False)

if __name__ == "__main__":
    rando_test()
    # pp_test()