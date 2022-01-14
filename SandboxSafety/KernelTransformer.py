import numpy as np
from SandboxSafety.Simulator.Dynamics import update_complex_state


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

    def __len__(self): return self.n_modes

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


class TestKernel:
    def __init__(self, conf):
        self.kernel = np.load(f"{conf.kernel_path}ObsKernel_{conf.kernel_mode}.npy")

        self.plotting = False
        self.resolution = conf.n_dx
        self.m = Modes(conf)


    def check_state(self, state=[0, 0, 0, 0, 0]):
        i, j, k, m = self.get_indices(state)

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

def check_fast_state(kernel, state):
    resolution = 40 #TODO: manual set
    x_ind = min(max(0, int(round((state[0])*resolution))), kernel.shape[0]-1)
    y_ind = min(max(0, int(round((state[1])*resolution))), kernel.shape[1]-1)
    theta_ind = int(round((state[2] + np.pi/2) / np.pi * (kernel.shape[2]-1)))
    theta_ind = min(max(0, theta_ind), kernel.shape[2]-1)

    mode = self.m.get_mode_id(state[3], state[4])


def index_transformer(conf):
    m = Modes(conf)
    kernel = np.load(f"{conf.kernel_path}ObsKernel_{conf.kernel_mode}.npy")
    index_kernel = np.zeros_like(kernel)

    inds = np.nonzero(kernel)
    inds = np.array(inds).T
    length = inds.shape[0]
    

    turtle = np.zeros((length, len(m)))
    print(turtle.shape)

    kernel = TestKernel(conf)

    for i in range(length):
        idx = inds[i]
        index_kernel[idx[0], idx[1], idx[2], idx[3]] = i

        d, v = m.qs[idx[3]]
        state = np.array([idx[0], idx[1], idx[2], v, d])
        valid_window = simulate_and_classify(state, m.qs, kernel, time_step=0.1)

        turtle[i] = valid_window

        print(i)

    np.save(f"{conf.kernel_path}ObsIdxTurtle_{conf.kernel_mode}.npy", index_kernel)
    np.save(f"{conf.kernel_path}ObsTurtle_{conf.kernel_mode}.npy", turtle)

    print(f"Finished")

def simulate_and_classify(state, dw, kernel, time_step=0.1):
    valid_ds = np.ones(len(dw))
    for i in range(len(dw)):
        next_state = update_complex_state(state, dw[i], time_step)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

    return valid_ds 





import yaml 
from argparse import Namespace
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

if __name__ == "__main__":
    # kt = KernelTransformer(load_conf("forest_kernel"))

    index_transformer(load_conf("forest_kernel"))



