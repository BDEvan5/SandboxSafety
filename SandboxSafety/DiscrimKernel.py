import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SandboxSafety.Simulator.Dynamics import update_std_state, update_complex_state, update_complex_state_const
from SandboxSafety.KernelTests.GeneralTestTrain import load_conf

from SandboxSafety.KernelGenerator import prepare_track_img, BaseKernel



class DiscrimGenerator(BaseKernel):
    def __init__(self, track_img, sim_conf):
        super().__init__(track_img, sim_conf)
        
        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))

        self.kernel[:, :, :] = track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

        self.dynamics = build_discrim_dynamics(self.phis, self.qs, self.velocity, self.t_step, self.sim_conf)

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = discrim_loop(self.kernel, self.n_modes, self.dynamics)
            # self.view_build(False)
            # self.view_kernel(0, False, z)

        return self.get_filled_kernel()
        

    def view_kernel(self, phi, show=True, n=0):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, -1]
            # print(f"KernelDyns: Mode: {m} -> i, j: {di},{dj}")

            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)

        if show:
            plt.show()

    def make_kernel_img(self, phi, show=True, n=0):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        plt.pause(0.0001)

        plt.savefig(f"SandboxSafety/Kernels/Obs_build_{n}.svg")
        plt.xticks([])
        plt.yticks([])
        
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


# @njit(cache=True)
def build_discrim_dynamics(phis, qs, velocity, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    dynamics = np.zeros((len(phis), len(qs), 8, 3), dtype=np.int)
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
                state = np.array([0, 0, p, velocity, 0])
                action = np.array([m, velocity])
                new_state = update_complex_state(state, action, time)
                dx, dy, phi = new_state[0], new_state[1], new_state[2]

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi < -np.pi:
                    phi = phi + 2*np.pi

                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                temp_dynamics = generate_temp_dynamics(dx, dy, h, resolution)
                
                dynamics[i, j, :, 0:2] = np.copy(temp_dynamics)

                # if t == 4:
                #     print(f"State: {state}")
                #     print(f"Action: {action}")
                #     print(f"New state: {new_state}")
                #     print(f"Std new state: {std_new_state}")
                #     print(f"Difference: {ds}")
                #     print(temp_dynamics)
                #     print("------------------")
                #     pass

    return dynamics

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

    return temp_dynamics

@njit(cache=True)
def discrim_loop(kernel, n_modes, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                    if kernel[i, j, k] == 1:
                        continue 
                    kernel[i, j, k] = check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel)

    return kernel

@njit(cache=True)
def check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel):
    l_xs, l_ys, l_phis, = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        for n in range(dynamics.shape[2]):
            di, dj, new_k = dynamics[k, l, n, :]
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            if previous_kernel[new_i, new_j, new_k]:
                # if you hit a constraint, break
                safe = False # breached a limit.
                break
        if safe:
            return False

    return True



def build_track_discrim(conf):
    img = prepare_track_img(conf) 
    # plt.figure(1)
    # plt.imshow(img)
    # plt.pause(0.0001)
    kernel = DiscrimGenerator(img, conf)
    kernel.calculate_kernel(100)
    kernel.save_kernel(f"Kernel_disc_{conf.map_name}")
    kernel.view_build(True)


def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = DiscrimGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = DiscrimGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")


def construct_obs_track(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx* 1.2) 
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, obs_offset:obs_size+obs_offset] = 1 
    # kernel = ViabilityGenerator(img, conf)
    kernel = DiscrimGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.view_build(True)

    kernel.save_kernel(f"ObsKernelTrack_{conf.track_kernel_path}")


if __name__ == "__main__":
    conf = load_conf("track_kernel")
    # conf.map_name = "race_track"
    build_track_discrim(conf)
    # construct_obs_track(conf)

    # conf = load_conf("forest_kernel")
    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

