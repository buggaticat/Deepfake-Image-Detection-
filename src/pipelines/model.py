import torch
from torch import nn
import timm
from .load_data import load_config

class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )
    
    def forward(self, input_tensor):
        fft = torch.fft.fft2(input_tensor)
        fft = torch.abs(fft)
        fft = torch.log1p(fft)
        fft = fft / (torch.amax(fft, dim = (1, 2, 3), keepdim = True) + 1e-8)
        return self.cnn(fft)

class ViTBranch(nn.Module):
    def __init__(self, vit_model_name):
        super().__init__()
        self.model = timm.create_model(vit_model_name, pretrained = True)
        self.model.head = nn.Identity()
    
    def forward(self, input_tensor):
        return self.model(input_tensor)
    
class EfficientNetBranch(nn.Module):
    def __init__(self, efficientnet_model_name):
        super().__init__()
        self.model = timm.create_model(efficientnet_model_name, pretrained = True)
        self.model.classifier = nn.Identity()
    
    def forward(self, input_tensor):
        return self.model(input_tensor)

class DeepfakeDetector(nn.Module):
    def __init__(self, vit_model_name, efficientnet_model_name, dropout, fusion_hidden_size):
        super().__init__()
        self.fft = FFTBranch()
        self.vit = ViTBranch(vit_model_name)
        self.efficientnet = EfficientNetBranch(efficientnet_model_name)
        self.fusion_head = nn.Sequential(
            nn.Linear(3328, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vit_efficientnet_tensor, fft_tensor, clip_embedding):
        fft_branch = self.fft(fft_tensor)
        vit_branch = self.vit(vit_efficientnet_tensor)
        efficientnet_branch = self.efficientnet(vit_efficientnet_tensor)
        concat = torch.cat([vit_branch, efficientnet_branch, fft_branch, clip_embedding], dim = 1)
        return self.fusion_head(concat)    

def build_model(device):
    cfg = load_config("local")['model']
 
    vit_model_name          = cfg['vit_model_name']
    efficientnet_model_name = cfg['efficientnet_model_name']
    dropout                 = cfg['dropout']
    fusion_hidden_size      = cfg['fusion_hidden_size']
 
    model = DeepfakeDetector(vit_model_name, efficientnet_model_name, dropout, fusion_hidden_size)
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model  = build_model(device)

    # Sanity check — run a dummy batch through
    dummy_vit = torch.randn(2, 3, 224, 224).to(device)   # batch of 2 RGB images
    dummy_fft = torch.randn(2, 1, 224, 224).to(device)   # batch of 2 grayscale images
    dummy_clip = torch.randn(2, 512).to(device)           # batch of 2 CLIP embeddings
 
    with torch.no_grad():
        output = model(dummy_vit, dummy_fft, dummy_clip)
 
    print(f"Output shape: {output.shape}")   # should be (2, 1)
    print(f"Output values: {output}")        # should be between 0 and 1

#python src/pipelines/model.py