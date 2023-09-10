import os
import torch 
import nibabel as nib
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
        if filename.endswith('nii.gz'):
            file_path = os.path.join(folder_path, filename)
            nifti_image = nib.load(file_path)
            nifti_array = nifti_image.get_fdata()
            nifti_arrays.append(nifti_array)
    
    return nifti_arrays.toTensor

class TCGA_GBM(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    
