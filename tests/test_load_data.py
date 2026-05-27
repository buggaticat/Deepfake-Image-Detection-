import os
import pandas as pd
import pytest
from src.pipelines.load_data import get_and_materialize

@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

def test_get_and_materialize_balance(mock_stream, tmp_path):
    get_and_materialize(mock_stream, 10, [0.5, 0.5], "train", 0, data_root=str(tmp_path), seed=42)
    df = pd.read_csv("metadata_train.csv")
    assert len(df[df['label'] == 'real']) == 5
    assert len(df[df['label'] == 'fake']) == 5


def test_get_and_materialize_saves_images(mock_stream, tmp_path):
    get_and_materialize(mock_stream, 4, [0.5, 0.5], "train", 0, data_root=str(tmp_path), seed=42)
    real_dir = os.path.join(str(tmp_path), "train", "real")
    fake_dir = os.path.join(str(tmp_path), "train", "fake")
    assert os.path.isdir(real_dir) 
    assert os.path.isdir(fake_dir)
    assert (len(os.listdir(real_dir)) + len(os.listdir(fake_dir))) == 4

def test_get_and_materialize_csv_output(mock_stream, tmp_path):
    get_and_materialize(mock_stream, 4, [0.5, 0.5], "train", 0, data_root=str(tmp_path), seed=42)
    df = pd.read_csv("metadata_train.csv")
    expected_columns = {'file_path', 'label', 'model', 'prompt'}
    assert set(df.columns) == expected_columns
    assert len(df.columns) == 4
    assert set(df['label'].unique()) == {'real', 'fake'}

def test_get_and_materialize_resume(mock_stream, tmp_path):
    get_and_materialize(mock_stream, 4, [0.5, 0.5], "train", 0, data_root=str(tmp_path), seed=42)
    
    real_dir = os.path.join(str(tmp_path), "train", "real")
    first_file = os.path.join(real_dir, os.listdir(real_dir)[0])
    mtime_before = os.path.getmtime(first_file)
 
    get_and_materialize(mock_stream, 4, [0.5, 0.5], "train", 0, data_root=str(tmp_path), seed=42)
    mtime_after = os.path.getmtime(first_file)
 
    assert mtime_before == mtime_after

#pytest tests/test_load_data.py