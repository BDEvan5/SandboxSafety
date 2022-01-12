from SandboxSafety.DiscrimKernel import DiscrimGenerator
from SandboxSafety.ViabKernel import ViabilityGenerator

import numpy as np

"""
    External functions
"""

def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx * 1.1) # the 1.1 makes the obstacle slightly bigger to take some error into account.
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 

    if conf.kernel_mode == 'viab':
        kernel = ViabilityGenerator(img, conf)
    elif conf.kernel_mode == 'disc':
        kernel = DiscrimGenerator(img, conf)

    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_mode}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_mode}")
