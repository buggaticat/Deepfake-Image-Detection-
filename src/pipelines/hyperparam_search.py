import os
import gc
import yaml
import optuna
import torch
import functools
from torch.utils.data import Subset
from .load_data import load_config
from .dataset import DeepFakeDataset, get_dataloader
from .model import DeepfakeDetector
from .train import train_one_epoch, validate

def objective(trial, device, train_loader, val_loader, vit_model_name, efficientnet_model_name, n_epochs_per_trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    fusion_hidden_size = trial.suggest_categorical("fusion_hidden_size", [256, 512, 1024])
    
    try:
        model = DeepfakeDetector(vit_model_name, efficientnet_model_name, dropout, fusion_hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        loss_function = torch.nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device)

        for epoch in range(n_epochs_per_trial):
            train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device)
            auroc_score, _, _ = validate(model, val_loader, loss_function, device)
            trial.report(auroc_score.item(), epoch + 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    finally:
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'scaler' in locals(): del scaler

        gc.collect()
        torch.cuda.empty_cache()

    return auroc_score.item()


def run_search():
    cfg = load_config("local")

    cfg_hyperparam_search = cfg['hyperparam_search']
    cfg_dataset           = cfg['dataset']
    cfg_preprocess        = cfg['preprocess']
    cfg_model             = cfg['model']

    seed = cfg['data']['seed']
 
    image_size           = cfg_preprocess['image_size']
    mean                 = cfg_preprocess['mean']
    std                  = cfg_preprocess['std']
    num_output_channels  = cfg_preprocess['num_output_channels']
    train_csv_path       = cfg_preprocess['csv_path']['train']
    validation_csv_path  = cfg_preprocess['csv_path']['validation']

    batch_size             = cfg_dataset['batch_size']
    num_workers            = cfg_dataset['num_workers']
    clip_model_name        = cfg_dataset['clip_model_name']
    preprocessed_data_root = cfg_dataset['preprocessed_data_root']
    
    n_trials =  cfg_hyperparam_search['n_trials']
    n_epochs_per_trial = cfg_hyperparam_search['n_epochs_per_trial']
    train_size = cfg_hyperparam_search['train_size']
    val_size = cfg_hyperparam_search['val_size']

    vit_model_name = cfg_model['vit_model_name']
    efficientnet_model_name = cfg_model['efficientnet_model_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DeepFakeDataset(
        train_csv_path, clip_model_name, preprocessed_data_root, "train", device,
        image_size, mean, std, num_output_channels
    )
    validation_dataset = DeepFakeDataset(
        validation_csv_path, clip_model_name, preprocessed_data_root, "validation", device,
        image_size, mean, std, num_output_channels
    )

    torch.manual_seed(seed)
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    val_indices   = torch.randperm(len(validation_dataset))[:val_size]

    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(validation_dataset, val_indices)

    train_loader      = get_dataloader(train_subset, batch_size, num_workers, shuffle=True)
    validation_loader = get_dataloader(val_subset, batch_size, num_workers, shuffle=False)

    storage_path =  "sqlite:///outputs/hyperparam/database/optuna_search.db"
    os.makedirs("outputs/hyperparam/database", exist_ok=True)

    study = optuna.create_study(
        study_name = "Hyperparameter Search",
        storage = storage_path,
        load_if_exists = True,
        direction="maximize",         # we want to maximize AUROC
        pruner=optuna.pruners.MedianPruner()  # kills bad trials early
    )
    objective_fn = functools.partial(objective, device=device, train_loader=train_loader, val_loader=validation_loader, 
                                     vit_model_name=vit_model_name, efficientnet_model_name=efficientnet_model_name, 
                                     n_epochs_per_trial=n_epochs_per_trial)
    study.optimize(objective_fn, n_trials=n_trials)

    print(f"Best AUROC: {study.best_value}")
    print(f"Best params: {study.best_params}")

    cfg['model']['dropout']            = float(study.best_params['dropout'])
    cfg['model']['fusion_hidden_size'] = int(study.best_params['fusion_hidden_size'])

    cfg['train']['learning_rate'] = float(study.best_params['lr'])
    cfg['train']['weight_decay']  = float(study.best_params['weight_decay'])

    config_path = os.path.join("config", "local.yaml")

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    os.makedirs("outputs/hyperparam/results", exist_ok=True)
    optuna.visualization.plot_optimization_history(study).write_html("outputs/hyperparam/results/optimization_history.html")
    optuna.visualization.plot_param_importances(study).write_html("outputs/hyperparam/results/param_importances.html")
    print("Plots saved → outputs/hyperparam/results")

if __name__ == "__main__":
    run_search()
    

#python src/pipelines/hyperparam_search.py