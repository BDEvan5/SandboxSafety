from GeneralTestTrain import test_kernel_vehicle, load_conf

from SandboxSafety.Simulator.ForestSim import ForestSim
from SandboxSafety.KernelGenerator import construct_obs_kernel, construct_kernel_sides
from SandboxSafety.NavAgents.SimplePlanners import RandomPlanner, PurePursuit 
from SandboxSafety.SupervisorySystem import Supervisor, ForestKernel

import numpy as np
from matplotlib import pyplot as plt

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"
kernel_name = f"kernel_sap_{run_n}"



def rando_test():
    conf = load_conf("forest_kernel")

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = RandomPlanner()
    kernel = ForestKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 5, add_obs=True, wait=False)
    # test_kernel_vehicle(env, safety_planner, False, 100, wait=False)

def pp_test():
    conf = load_conf("forest_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    # kernel = ForestKernel(conf, True)
    kernel = ForestKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, False, 100, add_obs=True, wait=False)
    # test_kernel_vehicle(env, safety_planner, True, 100, add_obs=True, wait=False)

def test_kernels():

    conf = load_conf("forest_kernel")

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)

    env = ForestSim(conf)
    pp_planner = PurePursuit(conf)
    rando_planner = RandomPlanner()
    kernel = ForestKernel(conf)

    safety_planner = Supervisor(rando_planner, kernel, conf)
    test_kernel_vehicle(env, safety_planner, True, 5, add_obs=True, wait=False)

    safety_planner = Supervisor(pp_planner, kernel, conf)
    test_kernel_vehicle(env, safety_planner, True, 5, add_obs=True, wait=False)

def test_construction():
    conf = load_conf("forest_kernel")

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)




if __name__ == "__main__":
    # rando_test()
    pp_test()

    # test_kernels()

    # test_construction()