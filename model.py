from torch import nn
import torch 
import torch.nn.functional as F

class simpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=2000, z_dim=200):
        super().__init__()
        
        #encoding layers
        self.input_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_sigma = nn.Linear(hidden_dim, z_dim)
        
        #decoding layers
        self.z_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_output= nn.Linear(hidden_dim, input_dim)
        
        self.leakyReLU = nn.LeakyReLU(0.01)
    
    def encode(self, x):
        hidden = self.leakyReLU(self.input_hidden(x))
        mu, sigma = self.hidden_mu(hidden), self.hidden_sigma(hidden)
        return mu, sigma 
    
    def decode(self, z):
        hidden = self.leakyReLU(self.z_hidden(z))
        output = self.hidden_output(hidden)
        return torch.sigmoid(output)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu+ sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma 
    
if __name__ == "__main__":
    x = torch.randn(4, 784) #input dim: 28*28 = 784
    vae = simpleVAE(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape)
    print(mu.shape)
    print(sigma.shape)
        
        