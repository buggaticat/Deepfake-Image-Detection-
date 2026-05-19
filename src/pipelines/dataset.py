import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPModel
from run_load_data import load_config

LABEL_MAP = {"real": 0, "fake": 1}


class DeepFakeDataset(Dataset):
    def __init__(self, csv_path, clip_model_name, data_root, split, device):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.split = split
        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()  # inference mode, no gradients needed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        prompt = row['prompt']

        # Load vit/efficientnet preprocessed tensor
        vit_efficientnet_tensor_path = os.path.join(self.data_root, f"vit_efficientnet_{self.split}", label, f"preprocessed_{idx}.pt")
        vit_efficientnet_tensor = torch.load(vit_efficientnet_tensor_path, weights_only=True)

        # Load fft preprocessed tensor
        fft_tensor_path = os.path.join(self.data_root, f"fft_{self.split}", label, f"preprocessed_{idx}.pt")
        fft_tensor = torch.load(fft_tensor_path, weights_only=True)

        # Tokenize and encode with CLIP
        token_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77          # CLIP's max token length
        ).to(self.device)

        with torch.no_grad():
            clip_embedding = self.clip_model.get_text_features(**token_ids).squeeze(0)  # (512,)

        # Encode label
        label_tensor = torch.tensor(LABEL_MAP[label], dtype=torch.float32)

        return vit_efficientnet_tensor, fft_tensor, clip_embedding, label_tensor


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    cfg = load_config("local")['dataset']

    batch_size          = cfg['batch_size']
    num_workers         = cfg['num_workers']
    train_csv_path      = cfg['csv_path']['train']
    validation_csv_path = cfg['csv_path']['validation']
    test_csv_path       = cfg['csv_path']['test']
    clip_model_name     = cfg['clip_model_name']
    data_root           = cfg['data_root']
    device              = cfg['device']

    train_dataset      = DeepFakeDataset(train_csv_path, clip_model_name, data_root, "train", device)
    validation_dataset = DeepFakeDataset(validation_csv_path, clip_model_name, data_root, "validation", device)
    test_dataset       = DeepFakeDataset(test_csv_path, clip_model_name, data_root, "test", device)

    train_loader      = get_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    validation_loader = get_dataloader(validation_dataset, batch_size, num_workers, shuffle=False)
    test_loader       = get_dataloader(test_dataset, batch_size, num_workers, shuffle=False)

    # Quick sanity check — print one batch shape
    vit_batch, fft_batch, clip_batch, label_batch = next(iter(train_loader))
    print(f"ViT/EfficientNet tensor: {vit_batch.shape}")
    print(f"FFT tensor:              {fft_batch.shape}")
    print(f"CLIP embedding:          {clip_batch.shape}")
    print(f"Labels:                  {label_batch.shape}")