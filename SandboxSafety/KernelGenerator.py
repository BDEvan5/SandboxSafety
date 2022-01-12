import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SandboxSafety.Simulator.Dynamics import update_std_state, update_complex_state, update_complex_state_const

class BaseKernel:
    def __init__(self, track_img, sim_conf):
        self.velocity = 2 #TODO: make this a config param
        self.track_img = track_img
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.phi_range = sim_conf.phi_range
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = sim_conf.n_modes
        self.sim_conf = sim_conf
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = np.linspace(-self.max_steer, self.max_steer, self.n_modes)

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
        
        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        
        self.kernel[:, :, :] = self.track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

        self.dynamics = build_viability_dynamics(self.phis, self.qs, self.velocity, self.t_step, self.sim_conf)

    def view_kernel(self, phi, show=True, fig_n=1):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(fig_n)
        plt.clf()
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, -1]


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

            # self.view_build(False)
        return self.get_filled_kernel()

# @njit(cache=True)
def build_viability_dynamics(phis, qs, velocity, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range

    dynamics = np.zeros((len(phis), len(qs), 3), dtype=np.int)
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
                state = np.array([0, 0, p, velocity, 0])
                action = np.array([m, velocity])
                new_state = update_complex_state(state, action, time)
                dx, dy, phi = new_state[0], new_state[1], new_state[2]

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, 0] = int(round(dx * resolution))                  
                dynamics[i, j, 1] = int(round(dy * resolution))                  
                

    return dynamics

@njit(cache=True)
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                if kernel[i, j, k] == 1:
                    continue 
                kernel[i, j, k] = check_viable_state(i, j, k, dynamics, previous_kernel)

    return kernel

@njit(cache=True)
def check_viable_state(i, j, k, dynamics, previous_kernel):
    l_xs, l_ys, l_phis = previous_kernel.shape
    n_modes = dynamics.shape[1]
    for l in range(n_modes):
        di, dj, new_k = dynamics[k, l, :]
        new_i = min(max(0, i + di), l_xs-1)  
        new_j = min(max(0, j + dj), l_ys-1)

        if not previous_kernel[new_i, new_j, new_k]:
            return False
    return True

"""
    External functions
"""

def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx * 1.1)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")






# if __name__ == "__main__":
#     conf = load_conf("track_kernel")
#     build_track_kernel(conf)


