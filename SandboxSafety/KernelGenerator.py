from SandboxSafety.DiscrimKernel import DiscrimGenerator
from SandboxSafety.ViabKernel import ViabilityGenerator

from SandboxSafety.Simulator.Dynamics import update_complex_state

import numpy as np
from matplotlib import pyplot as plt

"""
    External functions
"""

def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx * 1.2) # the 1.1 makes the obstacle slightly bigger to take some error into account.
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 

    if conf.kernel_mode == 'viab':
        kernel = ViabilityGenerator(img, conf)
    elif conf.kernel_mode == 'disc':
        kernel = DiscrimGenerator(img, conf)
    else:
        raise ValueError(f"Unknown kernel mode: {conf.kernel_mode}")

    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_mode}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1

    if conf.kernel_mode == 'viab':
        kernel = ViabilityGenerator(img, conf)
    elif conf.kernel_mode == 'disc':
        kernel = DiscrimGenerator(img, conf)
    else:
        raise ValueError(f"Unknown kernel mode: {conf.kernel_mode}")
 
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_mode}")



class Modes:
    def __init__(self, sim_conf):
        self.nq_steer = sim_conf.nq_steer
        self.nq_velocity = sim_conf.nq_velocity
        self.max_steer = sim_conf.max_steer
        self.max_velocity = sim_conf.max_v
        self.min_velocity = sim_conf.min_v

        self.vs = np.linspace(self.min_velocity, self.max_velocity, self.nq_velocity)
        self.ds = np.linspace(-self.max_steer, self.max_steer, self.nq_steer)

        self.qs = None
        self.n_modes = None
        self.nv_modes = None
        self.v_mode_list = None
        self.nv_level_modes = None

        self.init_modes()

    def init_modes(self):
        b = 0.523
        g = 9.81
        l_d = 0.329

        mode_list = []
        v_mode_list = []
        nv_modes = [0]
        for i, v in enumerate(self.vs):
            v_mode_list.append([])
            for s in self.ds:
                if abs(s) < 0.06:
                    mode_list.append([s, v])
                    v_mode_list[i].append(s)
                    continue

                friction_v = np.sqrt(b*g*l_d/np.tan(abs(s))) *1.1 # nice for the maths, but a bit wrong for actual friction
                if friction_v > v:
                    mode_list.append([s, v])
                    v_mode_list[i].append(s)

            nv_modes.append(len(v_mode_list[i])+nv_modes[-1])

        self.qs = np.array(mode_list) # modes1
        self.n_modes = len(mode_list) # n modes
        self.nv_modes = np.array(nv_modes) # number of v modes in each level
        self.nv_level_modes = np.diff(self.nv_modes) # number of v modes in each level
        self.v_mode_list = v_mode_list # list of steering angles sorted by velocity

    def get_mode_id(self, v, d):
        # assume that a valid input is given that is within the range.
        v_ind = np.argmin(np.abs(self.vs - v))
        d_ind = np.argmin(np.abs(self.v_mode_list[v_ind] - d))
        
        return_mode = self.nv_modes[v_ind] + d_ind
        
        return return_mode


class TestKernel:
    def __init__(self, conf):
        self.kernel = np.load(f"{conf.kernel_path}ObsKernel_{conf.kernel_mode}.npy")

        self.plotting = False
        self.resolution = conf.n_dx
        self.m = Modes(conf)


    def check_state(self, state=[0, 0, 0, 0, 0]):
        i, j, k, m = self.get_indices(state)

        # print(f"Expected Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
        if self.plotting:
            self.plot_kernel_point(i, j, k, m)
        if self.kernel[i, j, k, m] != 0:
            return False # unsfae state
        return True # safe state

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.kernel.shape[2]-1)
        mode = self.m.get_mode_id(state[3], state[4])

        return x_ind, y_ind, theta_ind, mode


def kernel_tester(conf):
    """
    Write function to fire random points and then simulate a step and check that none of the random points end up inside the kernel.
    """
    kernel = TestKernel(conf)
    n_test = 10000

    np.random.seed(10)
    states = np.random.random((n_test, 5))
    states[:, 0] = states[:, 0]  * conf.forest_width * 0.8 + conf.forest_width * 0.1
    states[:, 1] = states[:, 1] *  conf.forest_width * 0.8
    states[:, 2] = states[:, 2] * np.pi - np.pi/2 # angles
    states[:, 3] = states[:, 3] * 4 + 2
    states[:, 4] = states[:, 4] * 0# 0.8 - 0.4
    actions = np.random.random((n_test, 2))
    actions[:, 0] = actions[:, 0] * 0.8 - 0.4
    actions[:, 1] = actions[:, 1] * 5 + 2

    unsafes = []

    for i in range(n_test):
        next_state = update_complex_state(states[i], actions[i], conf.time_step)
        safe = kernel.check_state(next_state)

        if not safe:
            print(f"UNSAFE --> State: {states[i]} -> Action: {actions[i]} -> Next State: {next_state}")
            # unsafes.append(i)

            plt.figure(1)
            plt.xlim([0, conf.forest_width])
            plt.ylim([0, conf.forest_width*2])
            plt.plot(states[i, 0], states[i, 1], 'x', markersize=16, color='b')
            plt.plot(next_state[0], next_state[1], 'x', markersize=16, color='r')

            plt.show()


import yaml 
from argparse import Namespace
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



if __name__ == "__main__":
    conf = load_conf("forest_kernel")

    # kernel_tester(conf)

