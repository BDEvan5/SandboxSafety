import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SandboxSafety.Simulator.Dynamics import update_std_state, update_complex_state, update_complex_state_const



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

        self.qs = np.array(mode_list)
        self.n_modes = len(mode_list)
        self.nv_modes = np.array(nv_modes)
        self.v_mode_list = np.array(v_mode_list)

        print(self.qs)
        print(v_mode_list)
        print(f"Number of modes: {self.n_modes}")
        print(f"Number of v modes: {nv_modes}")

    def get_mode_id(self, v, d):
        # assume that a valid input is given that is within the range.
        v_ind = np.argmin(np.abs(self.vs - v))
        d_ind = np.argmin(np.abs(self.v_mode_list[v_ind] - d))
        
        return_mode = self.nv_modes[v_ind] + d_ind
        
        return return_mode

    def __len__(self): return self.n_modes


class BaseKernel:
    def __init__(self, track_img, sim_conf):
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

    def save_kernel(self, name):
        np.save(f"{self.sim_conf.kernel_path}{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name}")

    def get_filled_kernel(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")
        return filled/total


class ViabilityGenerator(BaseKernel):
    def __init__(self, track_img, sim_conf):
        super().__init__(track_img, sim_conf)
        
        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        
        self.kernel[:, :, :, :] = self.track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))

        self.dynamics = build_viability_dynamics(self.phis, self.m, self.t_step, self.sim_conf)

    def view_kernel(self, phi, show=True, fig_n=1):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(fig_n)
        plt.clf()
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind, 0].T + self.o_map.T
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k, new_q = self.dynamics[phi_ind, m]


            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)
        if show:
            plt.show()
    
    def view_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
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

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, 4].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel speed: {2}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, 11].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel speed: {4}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, 15].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel speed: {6}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, 18].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {2}")

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

        self.view_speed_build(True)
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

@njit(cache=True)
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    for i in range(l_xs):
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
        di, dj, new_k1, new_q1 = dynamics[k, q, l, 0, :]
        new_i1 = min(max(0, i + di), l_xs-1)  
        new_j1 = min(max(0, j + dj), l_ys-1)

        di, dj, new_k2, new_q2 = dynamics[k, q, l, 1, :]
        new_i2 = min(max(0, i + di), l_xs-1)  
        new_j2 = min(max(0, j + dj), l_ys-1)

        if not previous_kernel[new_i1, new_j1, new_k1, new_q1] and not previous_kernel[new_i2, new_j2, new_k2, new_q2]:
            return False
    return True




from argparse import Namespace
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


def test_modes():
    conf = load_conf("forest_kernel")
    m = Modes(conf)
    m.init_modes()
    
    l = m.qs.copy()
    for q in l:
        print(f"{q} -> {m.get_mode_id(q[1], q[0])}")


if __name__ == "__main__":
    # test_q_fcns()
#     conf = load_conf("track_kernel")
#     build_track_kernel(conf)

    test_modes()
