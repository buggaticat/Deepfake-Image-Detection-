import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from tqdm import tqdm
from load_data import load_config
from dataset import DeepFakeDataset, get_dataloader
from model import build_model

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0

    for vit_tensor, fft_tensor, clip_embedding, label_tensor in tqdm(dataloader, desc = "Training"):
        vit_tensor = vit_tensor.to(device)
        fft_tensor = fft_tensor.to(device)
        clip_embedding = clip_embedding.to(device)
        label_tensor = label_tensor.to(device)

        optimizer.zero_grad() # sets gradient to 0
        
        with torch.amp.autocast(device_type = device):
            predictions = model(vit_tensor, fft_tensor, clip_embedding)
            batch_loss = criterion(predictions.squeeze(1), label_tensor) # squeezes predictions tensor to the same shape as label_tensor [x, 1] -> [x, ]
        
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += batch_loss.item() #to increase accuracy of value extracted, so we use item

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0

    auroc = torchmetrics.AUROC(task = "binary").to(device)
    f1 = torchmetrics.F1Score(task = "binary").to(device)

    with torch.no_grad():
        for vit_tensor, fft_tensor, clip_embedding, label_tensor in tqdm(dataloader, desc = "Validating"):
            vit_tensor = vit_tensor.to(device)
            fft_tensor = fft_tensor.to(device)
            clip_embedding = clip_embedding.to(device)
            label_tensor = label_tensor.to(device)
            
            predictions = model(vit_tensor, fft_tensor, clip_embedding)
            batch_loss = criterion(predictions.squeeze(1), label_tensor)
            
            auroc.update(predictions.squeeze(1), label_tensor)
            f1.update(predictions.squeeze(1), label_tensor)
            
            running_loss += batch_loss.item()
        
        auroc_score = auroc.compute()
        f1_score = f1.compute()
    
    return auroc_score, f1_score, running_loss / len(dataloader)

def train():
    cfg            = load_config("local")['dataset']
    cfg_preprocess = load_config("local")['preprocess']
    cfg_train      = load_config("local")['train']
 
    image_size           = cfg_preprocess['image_size']
    mean                 = cfg_preprocess['mean']
    std                  = cfg_preprocess['std']
    num_output_channels  = cfg_preprocess['num_output_channels']

    batch_size          = cfg['batch_size']
    num_workers         = cfg['num_workers']
    train_csv_path      = cfg['csv_path']['train']
    validation_csv_path = cfg['csv_path']['validation']
    clip_model_name     = cfg['clip_model_name']
    data_root           = cfg['data_root']

    learning_rate   = cfg_train['learning_rate']
    weight_decay    = cfg_train['weight_decay']
    num_epochs      = cfg_train['num_epochs']
    checkpoint_dir  = cfg_train['checkpoint_dir']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = DeepFakeDataset(
        train_csv_path, clip_model_name, data_root, "train", device,
        image_size, mean, std, num_output_channels
    )
    validation_dataset = DeepFakeDataset(
        validation_csv_path, clip_model_name, data_root, "validation", device,
        image_size, mean, std, num_output_channels
    )

    train_loader      = get_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    validation_loader = get_dataloader(validation_dataset, batch_size, num_workers, shuffle=False)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    loss_function = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer)
    scaler = torch.amp.GradScaler(device)
    best_val_loss = float('inf')
    best_auroc = 0.0

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        training_loss = train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device)
        auroc, f1, val_loss = validate(model, validation_loader, loss_function, device)
        
        scheduler.step(val_loss)

        print(f"Train loss:      {training_loss:.4f}")
        print(f"Val loss:        {val_loss:.4f}  (best: {best_val_loss:.4f})")
        print(f"AUROC:           {auroc:.4f}  (best: {best_auroc:.4f})")
        print(f"F1:              {f1:.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auroc    = auroc
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "auroc":                auroc,
                "f1":                   f1
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved → {checkpoint_path}")
 
        if auroc > best_auroc:
            best_auroc = auroc

if __name__ == "__main__":
    train()