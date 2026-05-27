import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPModel
from .load_data import load_config
from .preprocess import transform_img

LABEL_MAP = {"real": 0, "fake": 1}

class DeepFakeDataset(Dataset):
    def __init__(self, csv_path, clip_model_name, data_root, split, device, image_size, mean, std, num_output_channels):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.split = split
        self.device = device
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.num_output_channels = num_output_channels

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval() 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        row = self.df.iloc[idx]
        label = row['label']
        prompt = row['prompt'] if pd.notna(row['prompt']) else ""

        img = Image.open(row['file_path']).convert('RGB')
        vit_efficientnet_tensor = transform_img(img, "vit_efficientnet", self.image_size, self.mean, self.std, self.num_output_channels)
        
        fft_tensor = transform_img(img, "fft", self.image_size, self.mean, self.std, self.num_output_channels)

        token_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77          # CLIP's max token length
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**token_ids)
            if hasattr(text_features, 'pooler_output'):
                clip_embedding = text_features.pooler_output.squeeze(0)
            else:
                clip_embedding = text_features.squeeze(0)
            label_tensor = torch.tensor(LABEL_MAP[label], dtype=torch.float32)

        return vit_efficientnet_tensor, fft_tensor, clip_embedding, label_tensor


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )


if __name__ == "__main__":
    cfg = load_config("local")['dataset']
    cfg_preprocess = load_config("local")['preprocess']

    image_size           = cfg_preprocess['image_size']
    mean                 = cfg_preprocess['mean']
    std                  = cfg_preprocess['std']
    num_output_channels  = cfg_preprocess['num_output_channels']
    train_csv_path      = cfg_preprocess['csv_path']['train']
    validation_csv_path = cfg_preprocess['csv_path']['validation']
    test_csv_path       = cfg_preprocess['csv_path']['test']

    batch_size          = cfg['batch_size']
    num_workers         = cfg['num_workers']
    clip_model_name     = cfg['clip_model_name']
    preprocessed_data_root = cfg['preprocessed_data_root']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DeepFakeDataset(
        train_csv_path, clip_model_name, preprocessed_data_root, "train", device,
        image_size, mean, std, num_output_channels
    )
    validation_dataset = DeepFakeDataset(
        validation_csv_path, clip_model_name, preprocessed_data_root, "validation", device,
        image_size, mean, std, num_output_channels
    )
    test_dataset = DeepFakeDataset(
        test_csv_path, clip_model_name, preprocessed_data_root, "test", device,
        image_size, mean, std, num_output_channels
    )

    train_loader      = get_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    validation_loader = get_dataloader(validation_dataset, batch_size, num_workers, shuffle=False)
    test_loader       = get_dataloader(test_dataset, batch_size, num_workers, shuffle=False)

    # Quick sanity check — print one batch shape
    vit_batch, fft_batch, clip_batch, label_batch = next(iter(train_loader))
    print(f"ViT/EfficientNet tensor: {vit_batch.shape}")
    print(f"FFT tensor:              {fft_batch.shape}")
    print(f"CLIP embedding:          {clip_batch.shape}")
    print(f"Labels:                  {label_batch.shape}")

#python src/pipelines/dataset.py