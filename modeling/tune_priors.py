import optuna
import yaml
from pathlib import Path
from estimate_ate import load_config, load_data, fit_bayesian_scm, infer_ate
import numpy as np
import arviz as az
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective(trial):
    prior_mean = trial.suggest_float('prior_mean', 0.0, 1.0)
    prior_var = trial.suggest_float('prior_var', 0.1, 2.0)
    
    config = load_config()
    config['prior_mean'] = prior_mean
    config['prior_var'] = prior_var
    
    train_df = load_data('train')
    val_df = load_data('val')
    
    if train_df.empty:
        return np.inf
    
    model, trace = fit_bayesian_scm(train_df, config)
    val_ate, val_hdi = infer_ate(trace)
    
    hdi_width = val_hdi[1] - val_hdi[0]
    
    rhat_treat = az.rhat(trace, var_names=['beta_treat'])['beta_treat'].values
    if rhat_treat > 1.05:
        return hdi_width * 2  # Penalize poor convergence
    
    return hdi_width

def run_tuning():
    config = load_config()
    n_trials = config.get('n_trials', 10)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best priors:", study.best_params)
    print("Best HDI width:", study.best_value)
    
    best_config = load_config()
    best_config.update(study.best_params)
    output_filename = f"best_model_config_{n_trials}_trials.yaml"

    with open(f"configs/{output_filename}", "w") as f:
        yaml.dump({"model": best_config}, f)
    logging.info(f"Best config saved to configs/{output_filename}")

if __name__ == "__main__":
    run_tuning()