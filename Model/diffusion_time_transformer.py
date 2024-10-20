import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class TMSA(nn.Module):
    def __init__(self, time_emb_dim, num_heads, in_features):
        super().__init__()
        self.in_features = in_features
        self.mhsa = nn.MultiheadAttention(in_features, num_heads, bias = True, batch_first = True)
        self.proj_space = nn.ModuleList([
            nn.Linear(in_features, in_features), 
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features)])
        self.proj_time = nn.ModuleList([
            nn.Linear(time_emb_dim, in_features), 
            nn.Linear(time_emb_dim, in_features),
            nn.Linear(time_emb_dim, in_features)]) # In the future you could think of a better method to stack those layers

    def forward(self, x_s, x_t):
        # Needs to be modified to account for different dimensions between the space and time projections
        # There should be a third term in the sum if label embedding is used
        B, T, D, _ = x_s.size()
        x_s = x_s.view(B, D * D, T)
        q = self.proj_space[0](x_s) + self.proj_time[0](x_t).unsqueeze(1)
        k = self.proj_space[1](x_s) + self.proj_time[1](x_t).unsqueeze(1)
        v = self.proj_space[2](x_s) + self.proj_time[2](x_t).unsqueeze(1)

        qkv = self.mhsa(q, k, v)

        return qkv[0].reshape(B, self.in_features, D, D)
    
# Testing TMSA
    
x_t = torch.zeros((2, 512))
x_s = torch.zeros((2, 3, 64, 64))

# z = TMSA(256, 512, 4, 128)

# print(z(x_s, x_t).shape)

# sys.exit(0)

class DffiT_TransformerBlock(nn.Module):
    def __init__(self, input_dim, time_emb_dim, num_heads, in_features):
        super().__init__()
        self.tmsa = TMSA(time_emb_dim, num_heads, in_features)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_features, in_features, kernel_size=1),
            nn.GELU(),
        )
    
    def forward(self, x_s, x_t):
        x_s = x_s + self.tmsa(self.layernorm1(x_s), x_t)
        x_s = x_s + self.mlp(self.layernorm2(x_s))

        return x_s
    

# z = DffiT_TransformerBlock(input_dim=[16, 128, 32, 32], time_emb_dim=512, num_heads=4, in_features=128)

# print(z(x_s, x_t).shape)


class DiffiT_ResBlock(nn.Module):
    def __init__(self, input_dim, time_emb_dim, num_heads, in_features):
        super().__init__()
        self.in_features = in_features
        self.time_emb_dim = time_emb_dim
        self.input_dim = input_dim
        self.in_features = in_features
        self.DiffiT_TransformerBlock = DffiT_TransformerBlock(input_dim, time_emb_dim, num_heads, in_features)
        self.conv = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.groupnorm = nn.GroupNorm(1, in_features)
        self.silu = nn.SiLU()
    
    def forward(self, x_s, x_t):
        x_s = x_s + self.DiffiT_TransformerBlock(self.conv(self.silu(self.groupnorm(x_s))), x_t)
        
        return x_s
    
# z = DiffiT_ResBlock(input_dim=[16, 128, 32, 32], time_emb_dim=512, num_heads=4, in_features=128)

# print(z(x_s, x_t).shape)

    
class DiffiT_ResBlock_L(nn.Module):
    def __init__(self, L, input_dim, time_emb_dim, num_heads, in_features):
        super().__init__()
        self.L = L
        self.in_features = in_features
        self.time_emb_dim = time_emb_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.transformer_blocks = nn.ModuleList()
        for i in range(L):
            self.transformer_blocks.append(DiffiT_ResBlock(input_dim, time_emb_dim, num_heads, in_features))

    def forward(self, x_s, x_t):
        for i in range(self.L):
            x_s = self.transformer_blocks[i](x_s, x_t)

        return x_s

# z = DiffiT_ResBlock_L(L = 4, input_dim=[16, 128, 32, 32], time_emb_dim=512, num_heads=4, in_features=128)

# print(z(x_s, x_t).shape)

        
class UNET_DiffiT_Model(nn.Module):
    def __init__(self, in_channels, res_block_repeat, size_input, time_emb_dim, num_heads, starting_features):
        super().__init__()
        self.tokenizer = nn.Conv2d(in_channels, starting_features, kernel_size=1)
        self.downsample = nn.ModuleList(
            [nn.Conv2d(starting_features, starting_features * 2, kernel_size=3, stride=2, padding=1),
             nn.Conv2d(starting_features * 2, starting_features * 2, kernel_size=3, stride = 2, padding=1),
             nn.Conv2d(starting_features * 2, starting_features * 2, kernel_size=3, stride = 2, padding=1) 
             ]
        )
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(starting_features * 2, starting_features * 2, kernel_size=2, stride = 2),
            nn.ConvTranspose2d(starting_features * 2, starting_features * 2, kernel_size=2, stride = 2),
            nn.ConvTranspose2d(starting_features * 2, starting_features, kernel_size=2, stride = 2)
        ])
        self.projection = nn.ModuleList([
            nn.Conv2d(starting_features * 4, starting_features * 2, kernel_size=1),
            nn.Conv2d(starting_features * 4, starting_features * 2, kernel_size=1),
            nn.Conv2d(starting_features * 2, starting_features, kernel_size=1)
        ])
        self.head = nn.Sequential(
            nn.GroupNorm(1, starting_features),
            nn.Conv2d(starting_features, in_channels, kernel_size=1)
        )

        L = res_block_repeat
        
        self.ResTransformerBlocks = nn.ModuleList([
            DiffiT_ResBlock_L(L[0], [starting_features, size_input, size_input], time_emb_dim, num_heads, starting_features),
            DiffiT_ResBlock_L(L[1], [starting_features * 2, size_input // 2, size_input // 2], time_emb_dim, num_heads, starting_features * 2),
            DiffiT_ResBlock_L(L[2], [starting_features * 2, size_input // 4, size_input // 4], time_emb_dim, num_heads, starting_features * 2),
            DiffiT_ResBlock_L(L[3], [starting_features * 2, size_input // 8, size_input // 8], time_emb_dim, num_heads, starting_features * 2),
            DiffiT_ResBlock_L(L[2], [starting_features * 2, size_input // 4, size_input // 4], time_emb_dim, num_heads, starting_features * 2),
            DiffiT_ResBlock_L(L[1], [starting_features * 2, size_input // 2, size_input // 2], time_emb_dim, num_heads, starting_features * 2),
            DiffiT_ResBlock_L(L[0], [starting_features, size_input, size_input], time_emb_dim, num_heads, starting_features),
        ])

    def forward(self, x_s, x_t):
        x_s = self.tokenizer(x_s)

        # Encoder
        x_s1 = self.ResTransformerBlocks[0](x_s, x_t)
        x_s2 = self.downsample[0](x_s1)
        x_s2 = self.ResTransformerBlocks[1](x_s2, x_t)
        x_s3 = self.downsample[1](x_s2)
        x_s3 = self.ResTransformerBlocks[2](x_s3, x_t)
        x_s4 = self.downsample[2](x_s3)

        # Lowest resolution
        x_s4 = self.ResTransformerBlocks[3](x_s4, x_t)

        # Decoder
        x_s5 = self.projection[0](torch.cat((x_s3, self.upsample[0](x_s4)), dim = 1))
        x_s5 = self.ResTransformerBlocks[4](x_s5, x_t)
        x_s5 = self.projection[1](torch.cat((x_s2, self.upsample[1](x_s5)), dim = 1))
        x_s5 = self.ResTransformerBlocks[5](x_s5, x_t)
        x_s5 = self.projection[2](torch.cat((x_s1, self.upsample[2](x_s5)), dim = 1))
        x_s5 = self.ResTransformerBlocks[6](x_s5, x_t)

        return self.head(x_s5)
    

def laplace_noise_schedule(mu = 0.0, b = 0.5):
    lmb = lambda t: mu - b * torch.sign(torch.tensor(0.5 - t)) * torch.log(1 - 2 * torch.abs(torch.tensor(0.5 - t)))
    snr_func = lambda t: torch.exp(lmb(t))
    alpha_func = lambda t: torch.sqrt(snr_func(t) / (1 + snr_func(t)))
    sigma_func = lambda t: torch.sqrt(1 / (1 + snr_func(t)))

    return alpha_func, sigma_func

def precalculate_laplace(timesteps, alpha_func, sigma_func):
    alphas = [alpha_func(i / timesteps) for i in range(timesteps)]
    sigmas = [sigma_func(i / timesteps) for i in range(timesteps)]

    return torch.tensor(alphas, dtype=torch.float32), torch.tensor(sigmas, dtype=torch.float32)


def create_steps(steps, t_max, diff_time):
    T = t_max * diff_time
    division = [1]
    for i in range(2, steps + 1):
        division.append(np.floor(T / steps * i))

    return torch.tensor(division, dtype = torch.int)

def checkpoint(model, filepath):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, filepath):
    chkpt = torch.load(filepath)
    model.load_state_dict(chkpt["model_state_dict"])
    model.optimizer.load_state_dict(chkpt["optimizer_state_dict"])

class Diffusion_Model(nn.Module):
    def __init__(self, t_max,
                 steps,
                 checkpoint,
                 load_checkpoint,
                 save_model_filepath,
                 diffusion_time, 
                 time_emb_dim, 
                 res_block_repeat, 
                 in_channels, 
                 batch_size, 
                 input_dim, 
                 num_heads, 
                 starting_features):
        super().__init__()
        if t_max <= 0 or t_max > 0.99:
            raise ValueError("Input t_max is is not in the specified range")
        if steps <= 0 or steps > t_max * 1000:
            raise ValueError("input steps is not in the specified range")
        self.t_max = t_max
        self.steps = steps
        self.UNet = UNET_DiffiT_Model(in_channels, res_block_repeat, input_dim, time_emb_dim, num_heads, starting_features)
        self.mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU()
        )
        self.save_model_filepath = save_model_filepath
        self.checkpoint = checkpoint
        self.load_checkpoint = load_checkpoint
        self.mse = nn.MSELoss()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.seq = create_steps(steps, t_max, diffusion_time)
        self.alpha_func, self.sigmas_func = laplace_noise_schedule()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = 2e-4)
        self.alphas, self.sigmas = precalculate_laplace(diffusion_time, self.alpha_func, self.sigmas_func)
        self.diffusion_time = diffusion_time
        self.time_emb_dim = time_emb_dim

    def forward(self, x_s):
        t = torch.randint(low=1, high=1000, size=(self.batch_size, 1))

        noise = torch.randn_like(x_s)
        
        x_s = self.alphas[t].view(self.batch_size, 1, 1, 1) * x_s + self.sigmas[t].view(self.batch_size, 1, 1, 1) * noise

        t_emb = self.mlp(t.to(dtype = torch.float32))

        x_s_noise_pred = self.UNet(x_s, t_emb)

        loss = self.mse(x_s_noise_pred, noise)

        return loss

    def sample(self, batch_size):
        self.eval()
        with torch.no_grad():
            x_t = torch.randn((batch_size, self.in_channels, self.input_dim, self.input_dim))
            for t1, t2 in tqdm(zip(reversed(self.seq[:-1]), reversed(self.seq))):
                t_emb = self.mlp(torch.tensor([t2], dtype = torch.float32)).unsqueeze(0).repeat(batch_size, 1)

                prediction_noise = self.UNet(x_t, t_emb)

                bt1 = t1.repeat(batch_size)
                bt2 = t2.repeat(batch_size)

                if t2 > 1:
                    x_t = self.alphas[bt2].view(batch_size, 1, 1, 1) * (x_t - self.sigmas[bt1].view(batch_size, 1, 1, 1) * prediction_noise) / self.alphas[bt1].view(batch_size, 1, 1, 1) + \
                    self.sigmas[bt2].view(batch_size, 1, 1, 1) * prediction_noise
                else:
                    z = torch.rand_like(x_t) 

                    x_t = (x_t - self.sigmas[bt2].view(batch_size, 1, 1, 1) * prediction_noise) / self.alphas[bt2].view(batch_size, 1, 1, 1) + self.sigmas[bt2].view(batch_size, 1, 1, 1) * z

        self.train()

        return x_t


# Testing
    

# Training


model = Diffusion_Model(steps = 100, checkpoint=checkpoint, load_checkpoint=load_checkpoint, 
                        save_model_filepath="File location",
                        t_max = 0.99, diffusion_time=1000, time_emb_dim=256, 
                        res_block_repeat=[1, 1, 1, 1, 1, 1], in_channels=3, batch_size=16, 
                        input_dim=32, num_heads=1, starting_features=32)
