import torch 
from torch import nn, optim
from model import simpleVAE 
from torch.utils.data import DataLoader 
from plot import plot_loss 
from dataset import TCGA_GBM

def compute_epoch_loss(device, model, dataloader, loss_func):
    model.eval()
    current_loss, num_examples = 0., 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_reconstructed = model(x)
            loss = loss_func(x_reconstructed, x, reduction='sum')
            num_examples += x.size(0)
            current_loss += loss

        avg_loss = current_loss / num_examples
        return avg_loss

def train_simpleVAE(device, folder_path, batch_size=32, learning_rate = 0.0003, input_dim = 240*240*150, num_epochs=50, alpha = 1, plot_losses = True):
    
    dataset = TCGA_GBM(folder_path)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=0)
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    model = simpleVAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_func = nn.MSELoss()

    print(dataset.data.shape)
    model.train()
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(dataloader):
            print(x)
            x = torch.tensor(x)
            x= x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)

            loss = loss_func(x_reconstructed, x)
            kl_div = 0.5 * (sigma**2 + mu**2 -1 -torch.log(sigma))
            total_loss = alpha*loss + kl_div 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_dict['train_combined_loss_per_batch'].append(total_loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(loss.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
        
        epoch_loss = compute_epoch_loss(device, model, dataloader, loss_func)
        print('***Epoch: %03d/%03d | Loss: %.3f' % (i+1, num_epochs, epoch_loss))
        log_dict['train_combined_per_epoch'].append(epoch_loss.item())

    plot_loss(log_dict['train_reconstruction_loss_per_batch'], "Reconstruction loss per batch")
    plot_loss(log_dict['train_kl_loss_per_batch'], "KL loss per batch")
    plot_loss(log_dict['train_combined_loss_per_batch'], "Combined loss per batch")
    plot_loss(log_dict['train_combined_per_epoch'], "Combined loss per epoch")

    return log_dict()

            



