import torch
from unittest.mock import MagicMock, patch
import src.pipelines.dataset as dataset

@patch("src.pipelines.dataset.CLIPModel.from_pretrained")
@patch("src.pipelines.dataset.CLIPTokenizer.from_pretrained")
def test_dataset_initialization(mock_tokenizer_init, mock_clip_model_init, mock_dataset_csv, tmp_path):
    mock_tokenizer = MagicMock()
    mock_clip_model = MagicMock()
    mock_tokenizer_init.return_value = mock_tokenizer
    mock_clip_model_init.return_value = mock_clip_model
    mock_clip_model.to.return_value = mock_clip_model

    clip_model_name = "openai/clip-vit-base-patch32"
    data_root = str(tmp_path)
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mock_dataset = dataset.DeepFakeDataset(mock_dataset_csv, clip_model_name, data_root, "test", "cuda", image_size, mean, std, 1)

    assert len(mock_dataset) == 3
    assert mock_dataset.split == "test"
    assert mock_dataset.device == "cuda"
    assert mock_dataset.image_size == 224
    assert mock_dataset.num_output_channels == 1 
    mock_tokenizer_init.assert_called_once_with(clip_model_name)
    mock_clip_model_init.assert_called_once_with(clip_model_name)
    mock_clip_model.eval.assert_called_once()

@patch("src.pipelines.dataset.transform_img")
@patch("src.pipelines.dataset.Image.open")
@patch("src.pipelines.dataset.CLIPModel.from_pretrained")
@patch("src.pipelines.dataset.CLIPTokenizer.from_pretrained")
def test_dataset_getitem_mapping(mock_tokenizer_init, mock_clip_model_init, mock_image_open, 
                                 mock_transform_img, mock_dataset_csv, tmp_path, mock_image):
    mock_tokenizer_instance = MagicMock()
    mock_clip_model_instance = MagicMock()
    mock_tokenizer_init.return_value = mock_tokenizer_instance
    mock_clip_model_init.return_value = mock_clip_model_instance
    mock_clip_model_instance.to.return_value = mock_clip_model_instance
    mock_clip_model_instance.eval.return_value = mock_clip_model_instance

    mock_image_open.return_value = mock_image
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mock_vit_tensor = torch.randn(3, 224, 224)
    mock_fft_tensor = torch.randn(1, 224, 224)
    mock_transform_img.side_effect = [mock_vit_tensor, mock_fft_tensor]

    mock_tokenizer_output = MagicMock()
    mock_tokenizer_instance.return_value = mock_tokenizer_output
    mock_tokenizer_output.to.return_value = {"input_ids": torch.ones(1, 77, dtype=torch.long)}

    mock_text_features = MagicMock()
    mock_clip_model_instance.get_text_features.return_value = mock_text_features
    mock_text_features.pooler_output = torch.randn(1, 512)

    mock_dataset = dataset.DeepFakeDataset(
        mock_dataset_csv, "openai/clip-vit-base-patch32", str(tmp_path), 
        "train", "cuda", image_size, mean, std, 1
    )

    vit_res, fft_res, clip_res, label_res = mock_dataset[0]

    assert vit_res is mock_vit_tensor
    assert fft_res is mock_fft_tensor
    
    assert label_res.item() == 0.0  
    assert clip_res.shape == torch.Size([512])  

    mock_tokenizer_instance.assert_called_with(
        "a real authentic image",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )

@patch("src.pipelines.dataset.Image.open")
@patch("src.pipelines.dataset.CLIPModel.from_pretrained")
@patch("src.pipelines.dataset.CLIPTokenizer.from_pretrained")
def test_dataset_clip_attribute_fallback_handling(mock_tokenizer_init, mock_clip_model_init, mock_image_open,  
                                                  mock_dataset_csv, tmp_path, mock_image):  
    mock_tokenizer_instance = MagicMock()
    mock_clip_model_instance = MagicMock()
    mock_tokenizer_init.return_value = mock_tokenizer_instance
    mock_clip_model_init.return_value = mock_clip_model_instance
    mock_clip_model_instance.to.return_value = mock_clip_model_instance
    mock_clip_model_instance.eval.return_value = mock_clip_model_instance
    mock_image_open.return_value = mock_image

    raw_tensor_output = torch.randn(1, 512)
    mock_clip_model_instance.get_text_features.return_value = raw_tensor_output

    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mock_dataset = dataset.DeepFakeDataset(
        mock_dataset_csv, "openai/clip-vit-base-patch32", str(tmp_path), 
        "train", "cuda", image_size, mean, std, 1
    )

    _, _, clip_res, _ = mock_dataset[0]

    assert clip_res.shape == torch.Size([512])


@patch("src.pipelines.dataset.transform_img")
@patch("src.pipelines.dataset.Image.open")
@patch("src.pipelines.dataset.CLIPModel.from_pretrained")
@patch("src.pipelines.dataset.CLIPTokenizer.from_pretrained")
def test_dataloader_batch_collating(mock_tokenizer_init, mock_clip_model_init, mock_image_open, 
                                    mock_transform_img, mock_dataset_csv, tmp_path, mock_image):   
    mock_tokenizer_instance = MagicMock()
    mock_clip_model_instance = MagicMock()
    mock_tokenizer_init.return_value = mock_tokenizer_instance
    mock_clip_model_init.return_value = mock_clip_model_instance
    mock_clip_model_instance.to.return_value = mock_clip_model_instance
    mock_clip_model_instance.eval.return_value = mock_clip_model_instance
    mock_image_open.return_value = mock_image

    mock_transform_img.side_effect = [
        torch.randn(3, 224, 224),  
        torch.randn(1, 224, 224),
        torch.randn(3, 224, 224),  
        torch.randn(1, 224, 224)
    ]
    mock_text_features = MagicMock()
    mock_clip_model_instance.get_text_features.return_value = mock_text_features
    mock_text_features.pooler_output = torch.zeros(1, 512)

    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mock_dataset = dataset.DeepFakeDataset(
        mock_dataset_csv, "openai/clip-vit-base-patch32", str(tmp_path), 
        "train", "cuda", image_size, mean, std, 1
    )

    dataloader = dataset.get_dataloader(mock_dataset, batch_size=2, num_workers=0, shuffle=False)

    vit_batch, fft_batch, clip_batch, label_batch = next(iter(dataloader))

    assert vit_batch.shape == torch.Size([2, 3, 224, 224])
    assert fft_batch.shape == torch.Size([2, 1, 224, 224])
    assert clip_batch.shape == torch.Size([2, 512])
    assert label_batch.shape == torch.Size([2])

#pytest tests/test_dataset.py