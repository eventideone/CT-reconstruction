import torch.utils.data as data
import glob
import numpy as np

 
class my_data_loader(data.Dataset):
    def __init__(self, folder, iflow):
        self.low_fdk = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.clinical_fdk = sorted(glob.glob(folder + '/*fdk_clinical_dose_256.npy'))
        self.low_sino = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.low_sino = sorted(glob.glob(folder + '/*fdk_low_dose_256.npy'))
        self.target = sorted(glob.glob(folder + '/*clean_fdk_256.npy'))
        self.iflow = iflow

    def __len__(self):
        if self.iflow:
            return self.low_fdk.__len__()
        else:
            return self.clinical_fdk.__len__()

    def __getitem__(self, idx):
        if self.iflow:
            print(self.low_fdk[idx], self.target[idx])
            return np.load(self.low_fdk[idx]), np.load(self.target[idx])
        else:
            print(self.clinical_fdk[idx], self.target[idx])
            return np.load(self.clinical_fdk[idx]), np.load(self.target[idx])
