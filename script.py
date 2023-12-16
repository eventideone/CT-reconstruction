# %% Challenge imports

import torch
import numpy as np

##################################

# %% Your imports
from utils.loader import my_data_loader
from utils.proc import my_recon

##################################

# %% Load data
import os

folder = "npy"
iflow = True

# Provide a Dataloader functio or loop to iterate over files in folder

# e.g.

dataloader = my_data_loader(folder, iflow)

# You can assume that the data format, filenames and folder structure of the test will be the same as the training data.


# %% Main loop

# Either pytorch loop, or your own
for noisy_reconstruction, target_reconstruction in dataloader:
    reconstruction = my_recon(noisy_reconstruction, iflow)

##################################

# %% Notes:

# 1- Provide a Conda package. We can test it in Linux and Windows but we have limited GPU resources in Linux. Please specify the OS tested on.

# 2- Provide a README.md file with instructions on how to run your code.

# 3- Be available by email in the folliwing weeks. If we can't run your code we will contact you.

# 4- If you have any questions please contact us.

##################################
