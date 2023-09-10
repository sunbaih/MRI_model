import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_loss(loss_vector, label):
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(range(len(loss_vector)), (loss_vector), label=f'{label}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f"/users/bsun14/data/bsun14/Pipelines/cancer_models/tumors_VAE/{label}")
