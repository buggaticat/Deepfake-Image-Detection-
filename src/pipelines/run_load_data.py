import re
import subprocess
import sys
from .load_data import load_config, count_saved

def patch_load_data(load_data_path, split_name, idx):
    with open(load_data_path, "r") as f:
        content = f.read()

    # Match the block for this split, e.g:
    # ds_train = load_dataset(...).skip(808)
    # get_and_materialize(ds_train, ..., idx=807, idx_cnt=[434, 373], ...)

    # Patch the .skip() value
    content = re.sub(
        rf'(load_dataset\([^)]+split="{split_name}"[^)]*\)(?:\.skip\(\d+\))?)',
        lambda m: re.sub(r'\.skip\(\d+\)', '', m.group(0)) + (f'.skip({idx + 1})' if idx > 0 else ''),
        content
    )

    # Patch idx=
    content = re.sub(
        rf'(get_and_materialize\([^,]+{split_name}[^)]+idx\s*=\s*)\d+',
        rf'\g<1>{idx}',
        content
    )

    with open(load_data_path, "w") as f:
        f.write(content)

    print(f"  [runner] Patched {split_name}: skip={idx + 1}, idx={idx}")


def all_done(splits, data_root):
    for split_name, size in splits.items():
        real, fake = count_saved(data_root, split_name)
        if real + fake < size:
            return False
    return True


def main():
    cfg = load_config("local")['load_data']

    load_data_path = cfg['run_load_data']['load_data_path']

    data_root      = cfg['load_data']['data_root']
    splits         = cfg['load_data']['splits']

    run = 0
    while not all_done(splits, data_root):
        run += 1
        print(f"\n[runner] === Run #{run} ===")

        result = subprocess.run([sys.executable, load_data_path])

        # only reaches here if load_data.py crashed
        print(f"[runner] Crashed (exit code {result.returncode}), checking progress...")
        for split_name in splits:
            real, fake = count_saved(data_root, split_name)
            total = real + fake
            print(f"  {split_name}: {total} saved ({real} real, {fake} fake)")
            if total > 0:
                patch_load_data(load_data_path, split_name, total)

        print("[runner] Restarting load_data.py with updated offsets...")

if __name__ == "__main__":
    main()


#python src/pipelines/run_load_data.py