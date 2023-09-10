import dataset
from dataset import read_nifti_data, TCGA_GBM
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import simpleVAE
from train import train_simpleVAE


print("yay")

folder_path = "/Users/baihesun/cancer_data/TCGA-GBM_all_niftis"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32  # Specify your desired batch size
learning_rate = 0.0003 
shuffle = True   # Set to True if you want to shuffle the data
input_dim = 240*240*150
num_epochs = 5
alpha = 1 


dataset = TCGA_GBM(folder_path)
print(dataset.data.shape)







