import torch
from torch import nn
from torch.distributions import 

class FFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = nn.Sequential(
            torch.fft.rfft2(),
            torch.fft.fftshift(),
            torch.abs(),
            torch.log1p()
        )
    def forward(self, input_tensor):
        return self.fft(input_tensor)
    
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        return 