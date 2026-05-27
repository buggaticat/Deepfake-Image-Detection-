import os
import yaml
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def load_config(proj_status):
    proj_root = Path(__file__).resolve().parent.parent.parent
    config_path = os.path.join(proj_root, "config", f"{proj_status}.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def count_saved(data_root, split_name):
    real_dir = os.path.join(data_root, split_name, "real")
    fake_dir = os.path.join(data_root, split_name, "fake")
    real = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
    fake = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
    return real, fake

def get_and_materialize(ds, size, probabilities, split_name, idx, data_root="data", seed=42):
    n_real = int(size * probabilities[0])
    n_fake = size - n_real
    real, fake = count_saved(data_root, split_name)
    counts = {'real': real, 'fake': fake}
    metadata = []
    i = idx

    print(f"  Collecting {n_real} real + {n_fake} fake for '{split_name}'...")

    pbar = tqdm(total=size, initial = idx, desc=split_name, unit="img")
    ds_iter = iter(ds)

    while i <= size:
        try:
            example = next(ds_iter)  # TIFF crash happens here, now caught
        except StopIteration:
            break
        except Exception as e:
            tqdm.write(f"  Skipping bad fetch: {e}")
            continue

        try:
            label = example['label']

            if label == 'real' and counts['real'] < n_real:
                counts['real'] += 1
            elif label == 'fake' and counts['fake'] < n_fake:
                counts['fake'] += 1
            else:
                continue

            save_dir = os.path.join(data_root, split_name, label)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{i}.png")

            if not os.path.exists(save_path):
                example['image'].convert('RGB').save(save_path, "PNG")

            metadata.append({
                "file_path": save_path,
                "prompt": example.get('prompt', None),
                "label": label,
                "model": example.get('model', None)
            })
            i += 1
            pbar.update(1)

            if i % 50 == 0:
                tqdm.write(f"  [{split_name}] {i}/{size} saved | real: {counts['real']}/{n_real} | fake: {counts['fake']}/{n_fake}")

        except Exception as e:
            tqdm.write(f"  Skipping corrupted image: {e}")

    pbar.close()

    csv_path = f"metadata_{split_name}.csv"
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
    else:
        existing_df = pd.DataFrame()
    df = pd.DataFrame(metadata)
    df = pd.concat([existing_df, df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} records to {csv_path} | real: {counts['real']} fake: {counts['fake']}")


if __name__ == "__main__":
    cfg = load_config("local")

    dataset_name     = cfg['data']['dataset_name']
    subset           = cfg['data']['subset']
    data_root        = cfg['data']['data_root']
    seed             = cfg['data']['seed']
    probabilities    = list(cfg['data']['probabilities'].values())
    train_split      = cfg['data']['splits']['train']
    validation_split = cfg['data']['splits']['validation']
    test_split       = cfg['data']['splits']['test']

    print("Building train set...")
    ds_train = load_dataset(dataset_name, subset, split="train", streaming=True)
    get_and_materialize(ds_train, train_split, probabilities, "train", idx = 0, data_root=data_root, seed=seed)
    
    print("Building validation set...")
    ds_validation = load_dataset(dataset_name, subset, split="validation", streaming=True)
    get_and_materialize(ds_validation, validation_split, probabilities, "validation", idx = 0, data_root=data_root, seed=seed)

    print("Building test set...")
    ds_test = load_dataset(dataset_name, subset, split="test", streaming=True)
    get_and_materialize(ds_test, test_split, probabilities, "test", idx = 0, data_root=data_root, seed=seed)


#python src/pipelines/load_data.py