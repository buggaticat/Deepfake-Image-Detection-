import pytest

def test_config_data_section(cfg):
    data = cfg['data']
    
    assert isinstance(data['dataset_name'], str)
    assert isinstance(data['subset'], str)
    assert isinstance(data['data_root'], str)
    assert isinstance(data['seed'], int)
    assert isinstance(data['buffer_size'], int)
    assert isinstance(data['probabilities'], dict)
    assert isinstance(data['splits'], dict)
    
    assert data['buffer_size'] > 0, "Buffer size must be a positive integer!"
    
    probs = data['probabilities']
    assert isinstance(probs['real'], float)
    assert isinstance(probs['fake'], float)
    assert pytest.approx(probs['real'] + probs['fake']) == 1.0, "Probabilities must sum to 1.0!"
    assert 0.0 <= probs['real'] <= 1.0
    assert 0.0 <= probs['fake'] <= 1.0
    
    splits = data['splits']
    assert isinstance(splits['train'], int) and splits['train'] > 0
    assert isinstance(splits['validation'], int) and splits['validation'] > 0
    assert isinstance(splits['test'], int) and splits['test'] > 0


def test_config_preprocess_section(cfg):
    prep = cfg['preprocess']
    
    assert isinstance(prep['image_size'], int)
    assert isinstance(prep['mean'], list)
    assert isinstance(prep['std'], list)
    assert isinstance(prep['num_output_channels'], int)
    assert isinstance(prep['model_name'], str)
    assert isinstance(prep['csv_path'], dict)
    
    assert prep['image_size'] > 0
    assert prep['num_output_channels'] >= 1
    
    assert len(prep['mean']) == 3, "Mean normalization array must contain exactly 3 channels!"
    assert len(prep['std']) == 3, "Std normalization array must contain exactly 3 channels!"
    
    for m, s in zip(prep['mean'], prep['std']):
        assert isinstance(m, float)
        assert isinstance(s, float)
        
    csv = prep['csv_path']
    for split in ['train', 'validation', 'test']:
        assert isinstance(csv[split], str)
        assert csv[split].endswith('.csv'), f"The saved target file for {split} must use a .csv extension!"


def test_config_dataset_and_model_sections(cfg):
    ds = cfg['dataset']
    model = cfg['model']
    
    assert isinstance(ds['batch_size'], int)
    assert isinstance(ds['num_workers'], int)
    assert isinstance(ds['clip_model_name'], str)
    assert isinstance(ds['preprocessed_data_root'], str)
    
    assert ds['batch_size'] > 0, "Batch size must be greater than 0!"
    assert ds['num_workers'] >= 0, "Num workers cannot be a negative number!"
    
    assert isinstance(model['vit_model_name'], str)
    assert isinstance(model['efficientnet_model_name'], str)
    assert isinstance(model['dropout'], float)
    assert isinstance(model['fusion_hidden_size'], int)
    
    assert 0.0 <= model['dropout'] < 1.0, "Dropout rate must be a valid probability fraction [0, 1)!"
    assert model['fusion_hidden_size'] > 0


def test_config_train_and_evaluate_sections(cfg):
    train = cfg['train']
    eval_sec = cfg['evaluate']
    
    assert isinstance(train['learning_rate'], float)
    assert isinstance(train['weight_decay'], float)
    assert isinstance(train['num_epochs'], int)
    assert isinstance(train['checkpoint_dir'], str)
    
    assert train['learning_rate'] > 0.0, "Learning rate must be a positive float value!"
    assert train['weight_decay'] >= 0.0, "Weight decay cannot be negative!"
    assert train['num_epochs'] > 0
    
    assert isinstance(eval_sec['results_dir'], str)


def test_config_hyperparam_search_cross_dependencies(cfg):
    hp = cfg['hyperparam_search']
    data = cfg['data']

    assert isinstance(hp['n_trials'], int)
    assert isinstance(hp['n_epochs_per_trial'], int)
    assert isinstance(hp['train_size'], int)
    assert isinstance(hp['val_size'], int)
    
    assert hp['n_trials'] > 0
    assert hp['n_epochs_per_trial'] > 0
    
    assert hp['train_size'] <= data['splits']['train'], (
        f"Optuna search train_size ({hp['train_size']}) cannot exceed "
        f"available materialized dataset splits.train ({data['splits']['train']})!"
    )
    
    assert hp['val_size'] <= data['splits']['validation'], (
        f"Optuna search val_size ({hp['val_size']}) cannot exceed "
        f"available materialized dataset splits.validation ({data['splits']['validation']})!"
    )

#pytest tests/test_config.py