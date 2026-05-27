import pytest
import torch
import pandas as pd
import src.pipelines.preprocess as preprocess
from unittest.mock import MagicMock, patch


def test_transform_img_vit_efficientnet(mock_image):
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformed_img = preprocess.transform_img(mock_image, "vit_efficientnet", image_size, mean, std, 1)
    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (3, 224, 224)
    assert transformed_img[0, 0, 0].item() == pytest.approx(-2.1179, abs=1e-3)

def test_transform_img_fft(mock_image):
    image_size = 224
    mean = None
    std = None
    transformed_img = preprocess.transform_img(mock_image, "fft", image_size, mean, std, 1)
    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (1, 224, 224)

@patch("src.pipelines.preprocess.Image.open")
def test_generate_caption(mock_image_open, mock_image):
    mock_image_open.return_value = mock_image

    mock_blip_model = MagicMock()
    mock_blip_model.generate.return_value = [[101, 202, 303]]

    mock_processor = MagicMock()
    
    mock_processor_output = MagicMock()
    mock_processor.return_value = mock_processor_output
    
    dummy_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
    mock_processor_output.to.return_value = dummy_inputs

    mock_processor.decode.return_value = "a real photograph of a cat"

    result = preprocess.generate_caption(
        blip_model=mock_blip_model,
        processor=mock_processor,
        device="cuda",
        image_path="fake/path/to/image.png"
    )

    mock_image_open.assert_called_once_with("fake/path/to/image.png")

    mock_blip_model.generate.assert_called_once_with(
        pixel_values=dummy_inputs["pixel_values"], 
        max_new_tokens=50
    )

    mock_processor.decode.assert_called_once_with([101, 202, 303], skip_special_tokens=True)

    assert result == "a real photograph of a cat"

@patch("src.pipelines.preprocess.Image.open")
def test_fillna_prompt(mock_image_open, mock_image):
    mock_image_open.return_value = mock_image

    mock_blip_model = MagicMock()
    mock_blip_model.generate.return_value = [[101, 202, 303]]

    mock_processor = MagicMock()
    
    mock_processor_output = MagicMock()
    mock_processor.return_value = mock_processor_output
    
    dummy_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
    mock_processor_output.to.return_value = dummy_inputs

    mock_processor.decode.return_value = "a newly generated caption"

    result = preprocess.generate_caption(
        blip_model=mock_blip_model,
        processor=mock_processor,
        device="cuda",
        image_path="fake/path/to/image.png"
    )
    df = pd.DataFrame([{"file_path": "img1.png", "prompt": "a pre-existing caption"}, 
                       {"file_path": "img2.png", "prompt": None}])
    
    csv_path = "dummy.csv"

    preprocess.fillna_prompt(df, mock_blip_model, mock_processor, "cuda", csv_path)
    
    assert df.loc[0, 'prompt'] == "a pre-existing caption"
    assert df.loc[1, 'prompt'] == "a newly generated caption"

#pytest tests/test_preprocess.py