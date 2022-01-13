import numpy as np


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


class KernelTransformer:
    def __init__(self, conf):
        self.kernel = np.load(f"{conf.kernel_path}ObsKernel_{conf.kernel_mode}.npy")

        # self.m = Modes(conf)
        n_modes = 19
        s = self.kernel.shape
        # self.turtle = self.kernel.copy()[:, :, :, :, None]
        # print(self.turtle.shape)
        turtle =  np.ones((s[0], s[1], s[2], s[3], n_modes))
        print(turtle.shape)

        turtle[:, :, :, :, :] = self.kernel[:, :, :, :, None] * turtle

        print(turtle.shape)


import yaml 
from argparse import Namespace
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

if __name__ == "__main__":
    kt = KernelTransformer(load_conf("forest_kernel"))




