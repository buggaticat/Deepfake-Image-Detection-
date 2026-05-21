import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from load_data import load_config

def transform_img(img, model, image_size, mean, std, num_output_channels):
    if model == "vit_efficientnet":  
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif model == "fft":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_output_channels),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
    return transform(img)  

def generate_caption(blip_model, processor, device, image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def fillna_prompt(df, blip_model, processor, device, csv_path):
    df_na = df[df['prompt'].isna()]
    print(f"Filling {len(df_na)} missing prompts...")

    for i, (idx, filepath) in enumerate(tqdm(df_na['file_path'].items(), desc="Generating captions")):
        try:
            caption = generate_caption(blip_model, processor, device, filepath)
        except Exception as e:
            tqdm.write(f"Failed at {filepath}: {e}")
            caption = ""

        df.at[idx, 'prompt'] = caption

        if i % 50 == 0 and i > 0:
            df.to_csv(csv_path, index=False)
            tqdm.write(f"Checkpoint saved at {i}/{len(df_na)}")

    df.to_csv(csv_path, index=False)
    tqdm.write("All missing prompts have been filled!")


if __name__ == "__main__":
    cfg = load_config("local")

    csv_path_train      = cfg['preprocess']['csv_path']['train']
    csv_path_validation = cfg['preprocess']['csv_path']['validation']
    csv_path_test       = cfg['preprocess']['csv_path']['test']
    model_name          = cfg['preprocess']['model_name']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor  = BlipProcessor.from_pretrained(model_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    blip_model.eval()

    df_train = pd.read_csv(csv_path_train)
    fillna_prompt(df_train, blip_model, processor, device, csv_path_train)

    df_validation = pd.read_csv(csv_path_validation)
    fillna_prompt(df_validation, blip_model, processor, device, csv_path_validation)

    df_test = pd.read_csv(csv_path_test)
    fillna_prompt(df_test, blip_model, processor, device, csv_path_test)
  