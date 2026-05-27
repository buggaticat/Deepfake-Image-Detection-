import sys
import pytest
import pandas as pd
from pathlib import Path
from PIL import Image
import src.pipelines.load_data as load_data

PIPELINES_DIR = str(Path(__file__).resolve().parent.parent)
if PIPELINES_DIR not in sys.path:
    sys.path.insert(0, PIPELINES_DIR)

@pytest.fixture
def cfg():
    return load_data.load_config("local")

@pytest.fixture
def mock_stream():
    return [
        {"label": "real", "image": Image.new("RGB", (100, 100)), "model": "real_photo", "prompt": "a photo"} 
        for _ in range(20)
    ] + [
        {"label": "fake", "image": Image.new("RGB", (100, 100)), "model": "stable_diffusion", "prompt": "a fake photo"}
        for _ in range(20)
    ]

@pytest.fixture(autouse = True)
def change_test_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

@pytest.fixture
def mock_image():
    return Image.new("RGB", (400, 300))

@pytest.fixture
def mock_dataset_csv(tmp_path):
    csv_file = tmp_path / "test_split.csv"
    df = pd.DataFrame([
        {"file_path": "mock_img1.png", "label": "real", "prompt": "a real authentic image"},
        {"file_path": "mock_img2.png", "label": "fake", "prompt": "a synthetic deepfake artifact"},
        {"file_path": "mock_img3.png", "label": "fake", "prompt": None}  
    ])
    df.to_csv(csv_file, index=False)
    return str(csv_file)