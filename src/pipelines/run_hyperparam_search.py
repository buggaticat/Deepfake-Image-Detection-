import os
import sys
import subprocess
import optuna
from load_data import load_config

def get_no_trials(sql_storage_path, storage_path, study_name):    
    if os.path.exists(storage_path):
        study = optuna.load_study(study_name = study_name, storage = sql_storage_path)
        cur = len([t for t in study.trials if t.state.is_finished()])
    else:
        cur = 0
    
    return cur

if __name__ == "__main__":
    cfg = load_config("local")

    cfg_hyperparam_search = cfg['hyperparam_search']
    cfg_run_hyperparam_search = cfg['run_hyperparam_search']

    n_trials = cfg_hyperparam_search['n_trials']

    hyperparam_search_path = cfg_run_hyperparam_search['hyperparam_search']

    sql_storage_path = "sqlite:///outputs/results/optuna_search.db"
    storage_path = "outputs/results/optuna_search.db"
    
    cur = get_no_trials(sql_storage_path, storage_path, "Hyperparameter Search")
    run = 0

    while cur < n_trials:
        run += 1
        print(f"\n[runner] === Run #{run} ===")

        result = subprocess.run([sys.executable, hyperparam_search_path])

        # only reaches here if load_data.py crashed
        print(f"[runner] Crashed (exit code {result.returncode}), checking progress...")

        cur = get_no_trials(sql_storage_path, storage_path, "Hyperparameter Search")
        
        print("[runner] Restarting hyperparam_search.py with updated offsets...")
    
    