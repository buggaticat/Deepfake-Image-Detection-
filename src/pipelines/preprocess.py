import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from run_load_data import load_config

def transform_img(img, model, image_size, mean, std, num_output_channels):
    if model == "vit_efficientnet":  
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif model == "fft":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_output_channels),
            transforms.ToTensor()
        ])
    return transform(img)


def preprocess_image(df, model, image_size, mean, std, split_name, data_root, num_output_channels):
    for i, filepath in enumerate(tqdm(df['file_path'], desc=f"Preprocessing {model}")):
        label = df['label'].iloc[i]
        save_dir = os.path.join(data_root, model + "_" + split_name, label)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"preprocessed_{i}.pt")

        if os.path.exists(save_path):
            continue

        try:
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform_img(img, model, image_size, mean, std, num_output_channels)
            torch.save(img_tensor, save_path)
        except Exception as e:
            tqdm.write(f"Error processing file {filepath} at index {i}: {e}")


def generate_caption(qwen_model, processor, device, image_path):  
    image = Image.open(image_path).convert('RGB')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence. Focus on the subject, setting, and any notable visual details."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=100)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return caption.strip()


def fillna_prompt(df, qwen_model, processor, device, csv_path):
    df_na = df[df['prompt'].isna()]
    print(f"Filling {len(df_na)} missing prompts...")

    for i, (idx, filepath) in enumerate(tqdm(df_na['file_path'].items(), desc="Generating captions")): 
        try:
            caption = generate_caption(qwen_model, processor, device, filepath)
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

    image_size           = cfg['preprocess']['image_size']
    mean                 = cfg['preprocess']['mean']
    std                  = cfg['preprocess']['std']
    num_output_channels  = cfg['preprocess']['num_output_channels']
    data_root            = cfg['preprocess']['data_root']
    csv_path_train       = cfg['preprocess']['csv_path']['train']
    csv_path_validation  = cfg['preprocess']['csv_path']['validation']
    csv_path_test        = cfg['preprocess']['csv_path']['test']
    model_name           = cfg['preprocess']['model_name']
    device               = cfg['preprocess']['device']

    processor  = AutoProcessor.from_pretrained(model_name)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16  # halves VRAM usage
    ).to(device)
    qwen_model.eval()

    df_train = pd.read_csv(csv_path_train)
    preprocess_image(df_train, "vit_efficientnet", image_size, mean, std, "train", data_root, num_output_channels)
    preprocess_image(df_train, "fft", image_size, mean, std, "train", data_root, num_output_channels)
    fillna_prompt(df_train, qwen_model, processor, device, csv_path_train)

    df_validation = pd.read_csv(csv_path_validation)
    preprocess_image(df_validation, "vit_efficientnet", image_size, mean, std, "validation", data_root, num_output_channels)
    preprocess_image(df_validation, "fft", image_size, mean, std, "validation", data_root, num_output_channels)
    fillna_prompt(df_validation, qwen_model, processor, device, csv_path_validation)

    df_test = pd.read_csv(csv_path_test)
    preprocess_image(df_test, "vit_efficientnet", image_size, mean, std, "test", data_root, num_output_channels)
    preprocess_image(df_test, "fft", image_size, mean, std, "test", data_root, num_output_channels)
    fillna_prompt(df_test, qwen_model, processor, device, csv_path_test)