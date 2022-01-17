import numpy as np
from SandboxSafety.Modes import Modes
from SandboxSafety.Simulator.Dynamics import update_complex_state
from matplotlib import pyplot as plt

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

    def plot_kernel_point(self, i, j, k, m):
        plt.figure(6)
        plt.clf()
        plt.title(f"Kernel inds: {i}, {j}, {k}, {m}: {self.m.qs[m]}")
        img = self.kernel[:, :, k, m].T 
        plt.imshow(img, origin='lower')
        plt.plot(i, j, 'x', markersize=20, color='red')
        # plt.show()
        plt.pause(0.0001)

def kernel_tester(conf):
    """
    Write function to fire random points and then simulate a step and check that none of the random points end up inside the kernel.
    """
    kernel = TestKernel(conf)
    n_test = 10000
    resolution = conf.n_dx

    m = Modes(conf)

    # np.random.seed(10)
    # states = np.random.random((n_test, 5))
    # states[:, 0] = states[:, 0]  * conf.forest_width * 0.8 + conf.forest_width * 0.1
    # states[:, 1] = states[:, 1] *  conf.forest_width * 0.8
    # states[:, 2] = states[:, 2] * np.pi - np.pi/2 # angles
    # states[:, 3] = states[:, 3] * 4 + 2
    # states[:, 4] = states[:, 4] * 0# 0.8 - 0.4
    # actions = np.random.random((n_test, 2))
    # actions[:, 0] = actions[:, 0] * 0.8 - 0.4
    # actions[:, 1] = actions[:, 1] * 5 + 2

    # states = 
    inds = np.where(kernel.kernel == 0)
    states = np.array(inds).T # for all allowed states

    unsafes = 0
    for i in range(n_test):
        state = np.zeros(5)
        state[0:2] = states[i, 0:2] / resolution
        state[2] = states[i, 2] / conf.n_phi * np.pi - np.pi/2
        mode = m.qs[states[i, 3]] 
        state[3] = mode[1]
        state[4] = mode[0]

        # if not kernel.check_state(states[i]):
        #     continue # the initial state is not viable.

        state_safe = False
        for action in m.qs:
            next_state = update_complex_state(state, action, conf.lookahead_time_step)
            safe = kernel.check_state(next_state)

            if safe:
                state_safe = True
                break

        if not state_safe:
            unsafes += 1
            print(f"UNSAFE ({i}) --> State: {state} ")
            # unsafes.append(i)

            # plt.imshow(kernel.kernel[:, :, :, ])
            i, j, k, q = kernel.get_indices(next_state)
            kernel.plot_kernel_point(i, j, k, q)

            plt.figure(1)
            plt.xlim([0, conf.obs_img_size])
            plt.ylim([0, conf.obs_img_size])
            plt.plot(state[0], state[1], 'x', markersize=16, color='b')
            for action in m.qs:
                next_state = update_complex_state(state, action, conf.lookahead_time_step)
                print(f"Action: {action} -> Next State: {next_state}")
                plt.plot(next_state[0], next_state[1], 'x', markersize=16, color='r')

            # plt.show()
    
    print(f"Unsafes: {unsafes}")

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

    kernel_tester(conf)

