import os
import torch 
import SimpleITK as sitk

import numpy as np
from torch.utils.data import Dataset, DataLoader

def read_nifti_data(folder_path):
    """
    Parameters
    ----------
    folder_path : path to folder containing NIFTI data 

    Returns
    -------
    numpy array of NIfTI data

    """
    nifti_arrays = []
    
    os.chdir(folder_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(folder_path, filename)
            nifti_image = sitk.ReadImage(file_path)
            nifti_array = sitk.GetArrayFromImage(nifti_image)
            nifti_arrays.append(nifti_array)

    
    return torch.tensor(nifti_arrays)

class TCGA_GBM(Dataset):
    def __init__(self, data_path):
        data = read_nifti_data(data_path)
        self.data = data  #doing this to save time while debugging
        self.shape = list(data.size())
        self.length = self.shape[0]
        self._num_samples = self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    
