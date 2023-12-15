import torch.utils.data as data
import glob
import numpy as np


class my_data_loader(data.Dataset):
    def __init__(self, folder):
        self.low_fdk = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.clinical_fdk = sorted(glob.glob(folder + '/*fdk_clinical_dose_256.npy'))
        self.low_sino = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.low_sino = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.target = sorted(glob.glob(folder + '/*clean_fdk_256.npy'))
        if self.low_fdk.__len__() > 0:
            self.iflow = 1
        else:
            self.iflow = 0

    def __len__(self):
        if self.iflow:
            return self.low_fdk.__len__()
        else:
            return self.clinical_fdk.__len__()

    def __getitem__(self, idx):
        if self.iflow:
            return self.low_fdk[idx], self.target[idx]
        else:
            return self.clinical_fdk[idx], self.target[idx]
