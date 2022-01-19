
from SandboxSafety.Simulator.Dynamics import update_complex_state
from SandboxSafety.Modes import Modes

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image


def build_dynamics_table(sim_conf):
    m = Modes(sim_conf)
    phis = np.linspace(-sim_conf.phi_range/2, sim_conf.phi_range/2, sim_conf.n_phi)

    if sim_conf.kernel_mode == "viab":
        dynamics = build_viability_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
        np.save("viab_dyns.npy", dynamics)
    elif sim_conf.kernel_mode == 'disc':
        dynamics = build_disc_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
        np.save("disc_dyns.npy", dynamics)
    else:
        raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")


class KernelGenerator:
    def __init__(self, track_img, sim_conf, load_dyns=True):
        self.track_img = track_img
        self.sim_conf = sim_conf
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.phi_range = sim_conf.phi_range
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.m = Modes(sim_conf)
        self.n_modes = self.m.n_modes

        self.o_map = np.copy(self.track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        
        self.kernel[:, :, :, :] = self.track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))

        if load_dyns:
            self.dynamics = np.load(f"{self.sim_conf.kernel_mode}_dyns.npy")
        else:
            if sim_conf.kernel_mode == "viab":
                self.dynamics = build_viability_dynamics(self.phis, self.m, self.t_step, self.sim_conf)
                np.save("viab_dyns.npy", self.dynamics)
            elif sim_conf.kernel_mode == 'disc':
                self.dynamics = build_disc_dynamics(self.phis, self.m, self.t_step, self.sim_conf)
                np.save("disc_dyns.npy", self.dynamics)
            else:
                raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")

    def save_kernel(self, name):
        np.save(f"{self.sim_conf.kernel_path}{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name}")


        self.view_speed_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelSpeed_{name}_{self.sim_conf.kernel_mode}.png")

        self.view_angle_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelAngle_{name}_{self.sim_conf.kernel_mode}.png")



    def get_filled_kernel(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")
        return filled/total

    def view_angle_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        mode_ind = 6

        self.axs[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
     
    def view_speed_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)
        # phi_ind = 0
        # quarter_phi = int(len(self.phis)/4)
        # phi_ind = 

        inds = np.array([2, 6, 0, 5], dtype=int)

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, inds[0]].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel speed: {2}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, inds[1]].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel speed: {3}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, inds[2]].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel speed: {4}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, inds[3]].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel speed: {5}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
    
    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)


        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.sim_conf.kernel_path}Kernel_build_{self.sim_conf.kernel_mode}.svg")

        if show:
            plt.show()

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics)

            # self.view_kernel(0, False)
            self.view_speed_build(False)

        return self.get_filled_kernel()



# @njit(cache=True)
def build_viability_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range

    ns = 2

    dynamics = np.zeros((len(phis), len(m), len(m), ns, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time/2)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 0, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 0, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 0, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 0, 3] = int(new_q)                  
                

                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 1, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 1, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 1, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 1, 3] = int(new_q)                  
                

    return dynamics

def build_options(conf):
    b = 1 / (conf.n_dx) /2
    p = conf.phi_range / (conf.n_phi -1) /2
    v = (conf.max_v - conf.min_v) / (conf.nq_velocity - 1) /2
    s = 2 * conf.max_steer / (conf.nq_steer - 1) /2

    print(f"Limits: b: {b}, p: {p}, v: {v}, s: {s}")

    opt1 = np.array([[b,b,0, 0, 0]
        ,[b,-b,0, 0, 0]
        ,[-b,b,0, 0, 0]
        ,[-b,-b,0, 0, 0]])

    opt2 = np.array([[0,0,p, 0, 0]
                ,[0,0,-p, 0, 0]])

    opt3 = np.array([[0,0,0, v, s]
            ,[0,0,0, -v, s]
            ,[0,0,0, v, -s]
            ,[0,0,0, -v, -s]])

    file = open('ErrorOpts.txt', 'w')

    block_state = np.array([b, b, p, v, s])

    opts = []
    for o1 in opt1:
        for o2 in opt2:
            for o3 in opt3:
                state = o1 + o2 + o3
                opts.append(state)
                file.write(f"{state}\n")
    file.close()

    return np.array(opts)

# @njit(cache=True)
def build_disc_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    opts = build_options(conf)
    pts = len(opts)

    ns = 1
    invalid_counter = 0
    valid_couter = 0
    dynamics = np.zeros((len(phis), len(m), len(m), pts, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                
                new_states = get_new_states(state, opts, action, time)

                for l, new_state in enumerate(new_states):
                    dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                    new_q = m.get_safe_mode_id(vel, steer)

                    if new_q is None:
                        invalid_counter += 1
                        dynamics[i, j, k, :, :] = np.nan # denotes invalid transition
                        print(f"Invalid dyns: phi_ind: {i}, s_mode:{j}, action_mode:{k}")
                        continue
                    valid_couter += 1

                    while phi > np.pi:
                        phi = phi - 2*np.pi
                    while phi < -np.pi:
                        phi = phi + 2*np.pi

                    new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))

                    dynamics[i, j, k, l, 2] = min(max(0, new_k), len(phis)-1)
                    dynamics[i, j, k, l, 0] = int(round(dx * resolution))                  
                    dynamics[i, j, k, l, 1] = int(round(dy * resolution))                  
                    dynamics[i, j, k, l, 3] = int(new_q)
                
                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_safe_mode_id(vel, steer)
                # new_q = m.get_mode_id(vel, steer)


    print(f"Invalid counter: {invalid_counter} vs valid counter: {valid_couter}")
    print(f"Dynamics Table has been built: {dynamics.shape}")

    return dynamics

# def generate_discs(state, action, )


def get_new_states(state, opts, action, time):
    new_states = np.zeros((len(opts), 5))
    for i, opt in enumerate(opts):

        new_states[i] = update_complex_state(state+opt, action, time)
    
    return new_states


@njit(cache=True)
def generate_temp_dynamics(dx, dy, h, resolution):
    temp_dynamics = np.zeros((8, 2))

    for i in range(2):
        temp_dynamics[0 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[0 + i*4, 1] = int(round((dy -h) * resolution))
        temp_dynamics[1 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[1 + i*4, 1] = int(round((dy +h) * resolution))
        temp_dynamics[2 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[2 + i*4, 1] = int(round((dy +h )* resolution))
        temp_dynamics[3 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[3 + i*4, 1] = int(round((dy -h) * resolution))
    #TODO: this could just be 4 blocks. There is no phi discretisation going on here. Maybe
    #! this isn't workign
    return temp_dynamics


@njit(cache=True)
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    for i in range(l_xs):
        # if i == 150:
            # print(f"i: {i}")
        for j in range(l_ys):
            for k in range(l_phis):
                for q in range(l_qs):
                    if kernel[i, j, k, q] == 1:
                        continue 
                    kernel[i, j, k, q] = check_viable_state(i, j, k, q, dynamics, previous_kernel)

    return kernel


@njit(cache=True)
def check_viable_state(i, j, k, q, dynamics, previous_kernel):
    l_xs, l_ys, l_phis, n_modes = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        di, dj, new_k, new_q = dynamics[k, q, l, 0, :]
        # if np.isnan(new_q):
        #     safe = False
        #     continue # not a valid option. Don't even bother with safe = False
        if new_q == -9223372036854775808:
            continue

        for n in range(dynamics.shape[3]): # cycle through 8 block states
            di, dj, new_k, new_q = dynamics[k, q, l, n, :]

                # return True # not safe.
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            if previous_kernel[new_i, new_j, new_k, new_q]:
                # if you hit a constraint, break
                safe = False # breached a limit.
                break # try again and look for a new action

        if safe: # there exists a valid action
            return False # it is safe

    return True # it isn't safe because I haven't found a valid action yet...





"""
    External functions
"""

def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx * 1.2) # the 1.1 makes the obstacle slightly bigger to take some error into account.
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 

    kernel = KernelGenerator(img, conf, True)
    # kernel = KernelGenerator(img, conf, False)

    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_mode}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1

    kernel = KernelGenerator(img, conf, True)

    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_mode}")

from argparse import Namespace
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



def test_construction():
    conf = load_conf("forest_kernel")
    # build_dynamics_table(conf)

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)




if __name__ == "__main__":

    test_construction()

