import os
import torch
import torchmetrics
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns
from tqdm import tqdm              
from .load_data import load_config  
from .model import build_model
from .dataset import DeepFakeDataset, get_dataloader


def load_checkpoint(checkpoint_path, device):
    model = build_model(device)
    checkpoint_dct = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_dct['model_state_dict'])
    model.eval()
    print(f"Epoch: {checkpoint_dct['epoch']}")
    print(f"Validation loss: {checkpoint_dct['val_loss']:.4f}")
    return model

def evaluate(model, dataloader, device, results_dir):
    auroc      = torchmetrics.AUROC(task="binary").to(device)
    f1         = torchmetrics.F1Score(task="binary").to(device)
    accuracy   = torchmetrics.Accuracy(task="binary").to(device)
    precision  = torchmetrics.Precision(task="binary").to(device)
    recall     = torchmetrics.Recall(task="binary").to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(task="binary").to(device)

    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for vit_tensor, fft_tensor, clip_embedding, label_tensor in tqdm(dataloader, desc = "Evaluating"):
            
            vit_tensor     = vit_tensor.to(device)
            fft_tensor     = fft_tensor.to(device)
            clip_embedding = clip_embedding.to(device)
            label_tensor   = label_tensor.to(device)

            predictions = model(vit_tensor, fft_tensor, clip_embedding)
            
            auroc.update(predictions.squeeze(1), label_tensor)
            f1.update(predictions.squeeze(1), label_tensor)
            accuracy.update(predictions.squeeze(1), label_tensor)
            precision.update(predictions.squeeze(1), label_tensor)
            recall.update(predictions.squeeze(1), label_tensor)
            confusion_matrix.update(predictions.squeeze(1), label_tensor)

            all_predictions.append(predictions.squeeze(1).detach().cpu())
            all_labels.append(label_tensor.detach().cpu())
        
    auroc_score = auroc.compute()
    f1_score = f1.compute()
    accuracy_score = accuracy.compute()
    precision_score = precision.compute()
    recall_score = recall.compute()
    conf_matrix_result = confusion_matrix.compute()

    print("=== Test Results ===")
    print(f"Accuracy:  {accuracy_score:.4f}")
    print(f"AUROC:     {auroc_score:.4f}")
    print(f"F1:        {f1_score:.4f}")
    print(f"Precision: {precision_score:.4f}")
    print(f"Recall:    {recall_score:.4f}")

    all_predictions = torch.cat(all_predictions)
    all_labels      = torch.cat(all_labels)

    results_df = pd.DataFrame({
        "prediction":      all_predictions.cpu().numpy(),
        "label":           all_labels.cpu().numpy(),
        "predicted_label": ["fake" if p > 0.5 else "real" for p in all_predictions],
        "correct":         [(p > 0.5).item() == l.item() for p, l in zip(all_predictions, all_labels)]
    })

    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)
    print("Results saved")

    return conf_matrix_result

def plot_confusion_matrix(confusion_matrix, save_dir):
    if hasattr(confusion_matrix, "cpu"):
        confusion_matrix = confusion_matrix.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    labels = ["real", "fake"]

    sns.heatmap(confusion_matrix, annot = True, fmt = 'd', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    os.makedirs(save_dir, exist_ok=True)
    figure_save_path = os.path.join(save_dir, "confusion_matrix.png")
    
    plt.savefig(figure_save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    cfg_dataset = load_config("local")['dataset']
    cfg_preprocess = load_config("local")['preprocess']
    cfg_train = load_config("local")['train']
    cfg_evaluate = load_config("local")['evaluate']

    image_size           = cfg_preprocess['image_size']
    mean                 = cfg_preprocess['mean']
    std                  = cfg_preprocess['std']
    num_output_channels  = cfg_preprocess['num_output_channels']
    test_csv_path       = cfg_preprocess['csv_path']['test']

    clip_model_name     = cfg_dataset['clip_model_name']
    preprocessed_data_root = cfg_dataset['preprocessed_data_root']
    batch_size          = cfg_dataset['batch_size']
    num_workers         = cfg_dataset['num_workers']

    results_dir = cfg_evaluate['results_dir']

    checkpoint_path = os.path.join(cfg_train['checkpoint_dir'], "best_model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = DeepFakeDataset(
        test_csv_path, clip_model_name, preprocessed_data_root, "test", device,
        image_size, mean, std, num_output_channels
    )
    test_loader = get_dataloader(test_dataset, batch_size, num_workers, shuffle=False)

    model = load_checkpoint(checkpoint_path, device)
    confusion_matrix = evaluate(model, test_loader, device, results_dir)
    plot_confusion_matrix(confusion_matrix, results_dir)

#python src/pipelines/evaluate.py