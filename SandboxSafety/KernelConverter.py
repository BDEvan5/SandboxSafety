import numpy as np 
from SandboxSafety.Utils import load_conf
from numba import njit


def convert_kernel(sim_conf, opt):
    dynamics = np.load(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy")
    kernel = np.load(f"{sim_conf.kernel_path}{opt}Kernel_{sim_conf.kernel_mode}.npy")


    turtle, fish_tab = seaside_loop(kernel, dynamics)
    
    np.save(f"{sim_conf.kernel_path}{opt}Turtle_{sim_conf.kernel_mode}.npy", turtle)
    np.save(f"{sim_conf.kernel_path}{opt}FishTab_{sim_conf.kernel_mode}.npy", fish_tab)

    print(f"Process finished {opt}: fish tab length = {len(fish_tab)} out of {kernel.size}")



@njit(cache=True, parallel=True)
def seaside_loop(kernel, dynamics):
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    turtle = np.zeros_like(kernel)
    n_free = kernel.size - np.count_nonzero(kernel)
    fish_tab = np.zeros((n_free, l_qs))
    fish_idx = 1
    for i in range(l_xs):
        # print(f"{i}/{l_xs}")
        for j in range(l_ys):
            for k in range(l_phis):
                for q in range(l_qs):
                    if kernel[i, j, k, q] == 1:
                        turtle[i, j, k, q] = -2 # not valid at all
                        continue 
                    valid_window = calculate_valid_window(kernel, dynamics, i, j, k, q)
                    if not valid_window.any():
                        # print(f"Kernel Error: no opts -> {i}, {j}, {k}, {q}")
                        turtle[i, j, k, q] = -2
                        continue
                        # raise ValueError("Kernel not calculated properly")
                    if valid_window.all() == 1:
                        turtle[i, j, k, q] = -1 # all modes allowed
                        continue
                    
                    turtle[i, j, k, q] = int(fish_idx)
                    fish_tab[fish_idx] = valid_window
                    fish_idx += 1

    fish_tab = fish_tab[0:fish_idx+1]

    return turtle, fish_tab


@njit(cache=True)
def calculate_valid_window(kernel, dynamics, i, j, k, q):
    l_xs, l_ys, l_phis, n_modes = kernel.shape
    valid_window = np.zeros(n_modes)
    for l in range(n_modes):
        safe = True
        di, dj, new_k, new_q = dynamics[k, q, l, 0, :]
        if new_q == -9223372036854775808:
            valid_window[l] = 0
            continue

        for n in range(dynamics.shape[3]): # cycle through 8 block states
            di, dj, new_k, new_q = dynamics[k, q, l, n, :]

                # return True # not safe.
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            if kernel[new_i, new_j, new_k, new_q]:
                valid_window[l] = 0
                safe = False #
                break 

        if safe: # there exists a valid action
            valid_window[l] = 1 # action is safe

    return valid_window





if __name__ == "__main__":
    sim_conf = load_conf("forest_kernel")
    convert_kernel(sim_conf, "Obs")
    convert_kernel(sim_conf, "Side")
