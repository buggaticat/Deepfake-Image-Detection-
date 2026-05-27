import os
import pandas as pd
from PIL import Image
from load_data import load_config

def get_csv_path(split_name):
    csv_path = f"metadata_{split_name}.csv"
    return csv_path

def get_dataframe(csv_path):
    return pd.read_csv(csv_path)

def count_rows(split_name):
    csv_path = get_csv_path(split_name)
    return len(get_dataframe(csv_path))

def count_isna(split_name):
    df = get_dataframe(get_csv_path(split_name))
    fake_notna = len(df[(df['label'] == "fake") & (df['prompt'].notna())])
    fake_isna = len(df[(df['label'] == "fake") & (df['prompt'].isna())])
    real_notna = len(df[(df['label'] == "real") & (df['prompt'].notna())])
    real_isna = len(df[(df['label'] == "real") & (df['prompt'].isna())])
    return fake_notna, fake_isna, real_notna, real_isna


def audit_and_fix_split(split_name, data_root="data", csv_path=None):
    if csv_path is None:
        csv_path = f"metadata_{split_name}.csv"

    print(f"\n=== Auditing & Fixing {split_name} ===")

    if not os.path.exists(csv_path):
        print(f"  [!] No CSV found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  CSV entries (before): {len(df)}")

    # --- 1. COUNT ACTUAL FILES ON DISK ---
    real_dir = os.path.join(data_root, split_name, "real")
    fake_dir = os.path.join(data_root, split_name, "fake")
    real_files = set(os.listdir(real_dir)) if os.path.exists(real_dir) else set()
    fake_files = set(os.listdir(fake_dir)) if os.path.exists(fake_dir) else set()
    all_disk_paths = (
        {os.path.join(real_dir, f) for f in real_files} |
        {os.path.join(fake_dir, f) for f in fake_files}
    )
    print(f"  Disk files (before): {len(all_disk_paths)} ({len(real_files)} real, {len(fake_files)} fake)")

    csv_paths = set(df['file_path'].tolist())

    # --- 2. DELETE ORPHAN IMAGES (on disk but not in CSV) ---
    in_disk_not_csv = all_disk_paths - csv_paths
    print(f"  Deleting {len(in_disk_not_csv)} orphan images (on disk, not in CSV)...")
    for path in in_disk_not_csv:
        os.remove(path)
        print(f"    Deleted: {path}")

    # --- 3. DROP CSV ROWS WHERE FILE DOESN'T EXIST ON DISK ---
    in_csv_not_disk = csv_paths - all_disk_paths
    print(f"  Dropping {len(in_csv_not_disk)} CSV entries with no file on disk...")
    df = df[~df['file_path'].isin(in_csv_not_disk)]

    # --- 4. DROP DUPLICATE CSV ENTRIES (keep first) ---
    dupes_count = df.duplicated(subset='file_path', keep='first').sum()
    print(f"  Dropping {dupes_count} duplicate CSV entries...")
    df = df.drop_duplicates(subset='file_path', keep='first')

    # --- 5. CHECK AND DELETE CORRUPT IMAGES ---
    print(f"  Checking for corrupt images...")
    corrupt_paths = []
    for path in df['file_path'].tolist():
        if not os.path.exists(path):
            continue
        try:
            Image.open(path).verify()
        except Exception as e:
            corrupt_paths.append(path)
            print(f"    Corrupt: {path} — {e}")
            os.remove(path)
    print(f"  Deleted {len(corrupt_paths)} corrupt images.")
    df = df[~df['file_path'].isin(corrupt_paths)]

    # --- 6. SAVE CLEANED CSV ---
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV entries (after):  {len(df)}")

    real_remaining = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
    fake_remaining = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
    print(f"  Disk files (after):  {real_remaining + fake_remaining} ({real_remaining} real, {fake_remaining} fake)")

    print(f"\n  FIXED SUMMARY:")
    print(f"    Orphan images deleted:       {len(in_disk_not_csv)}")
    print(f"    CSV rows dropped (no file):  {len(in_csv_not_disk)}")
    print(f"    Duplicate CSV rows dropped:  {dupes_count}")
    print(f"    Corrupt images deleted:      {len(corrupt_paths)}")


if __name__ == "__main__":
    cfg_diagnose = load_config("local")['diagnose']
    
    splits = cfg_diagnose['splits']
    data_root = cfg[]

    audit_and_fix_split("train", data_root=DATA_ROOT)

    print("\n=== All splits cleaned ===")