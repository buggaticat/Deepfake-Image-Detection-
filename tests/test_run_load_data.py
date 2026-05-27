import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from src.pipelines.run_load_data import all_done, main, patch_load_data


def test_patch_load_data_modifies_code_correctly(tmp_path):
    """Verifies that regex updates both .skip() and idx parameters safely."""
    
    mock_script = os.path.join(tmp_path, "mock_load_data.py")
    mock_script = Path(mock_script)
    
    initial_code = (
        'print("Starting...")\n'
        'ds_train = load_dataset("fake", split="train", streaming=True).skip(10)\n'
        'get_and_materialize(ds_train, train_split, probabilities, "train", idx=9)\n'
    )
    mock_script.write_text(initial_code)

    patch_load_data(str(mock_script), split_name="train", idx=45)

    patched_code = mock_script.read_text()

    assert '.skip(46)' in patched_code
    assert 'idx=45' in patched_code
    assert '.skip(10)' not in patched_code
    assert 'idx=9' not in patched_code

def test_all_done_conditions():
    """Validates the logical completion checker wrapper."""
    splits = {"train": 100, "validation": 20}
    data_root = "dummy_root"

    with patch('src.pipelines.run_load_data.count_saved') as mock_count:
        mock_count.side_effect = lambda _, split: (25, 25) if split == "train" else (10, 10)
        assert all_done(splits, data_root) is False

        mock_count.side_effect = lambda _, split: (50, 50) if split == "train" else (10, 10)
        assert all_done(splits, data_root) is True

@patch("src.pipelines.run_load_data.load_config")
@patch("src.pipelines.run_load_data.subprocess.run")
@patch("src.pipelines.run_load_data.patch_load_data")
@patch("src.pipelines.run_load_data.count_saved")
@patch("src.pipelines.run_load_data.all_done")

def test_main_runner_execution_loop(mock_all_done, mock_count, mock_patch, mock_subprocess, mock_config):
    """Simulates a pipeline crash, progress logging, and a subsequent recovery restart."""
    
    mock_config.return_value = {
        'load_data': {
            'run_load_data': {'load_data_path': 'mock_load.py'},
            'load_data': {
                'data_root': 'data',
                'splits': {'train': 10}
            }
        }
    }

    mock_all_done.side_effect = [False, True]

    mock_count.return_value = (2, 2) 

    mock_process_result = MagicMock()
    mock_process_result.return_code = 1
    mock_subprocess.return_value = mock_process_result

    with patch.object(sys, 'argv', ['run_load_data.py']):
        main()

    mock_patch.assert_called_once_with('mock_load.py', 'train', 4)

#pytest tests/test_run_load_data.py
