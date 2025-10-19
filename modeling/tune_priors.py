# tune_priors.py
import optuna
import yaml
from pathlib import Path
from estimate_ate import load_config, load_data, fit_bayesian_scm, infer_ate
import numpy as np
import arviz as az
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective(trial):
    # Tune shared priors for beta_treat and beta_confound (vectors); extend to u_var if needed
    prior_mean = trial.suggest_float('prior_mean', -0.5, 0.5)  # Centered around lit-informed +0.05 for treat
    prior_var = trial.suggest_float('prior_var', 0.1, 2.0)
    u_var = trial.suggest_float('u_var', 1.0, 5.0)  # Latent U variance for unobserved confounders
    
    config = load_config()
    if config is None:
        logging.error("Config load failed; check path")
        return np.inf
    
    # Debug print: Confirm loaded values
    logging.info(f"Trial {trial.number}: Loaded n_draws={config.get('n_draws', 'MISSING')}, n_trials={config.get('n_trials', 'MISSING')}")
    
    train_df = load_data('train')
    
    if train_df.empty:
        return np.inf
    
    model, trace = fit_bayesian_scm(train_df, config)
    ate, hdi = infer_ate(trace)  # Use beta_treat HDI proxy (no test_df for tuning speed)
    
    hdi_width = hdi[1] - hdi[0]
    
    # Convergence check
    rhat_treat = az.rhat(trace, var_names=['beta_treat'])['beta_treat'].values
    try:
        rhat_confound = az.rhat(trace, var_names=['beta_confound'])['beta_confound'].mean().values
    except:
        rhat_confound = 1.1  # Fallback if vector issue
    if rhat_treat > 1.05 or rhat_confound > 1.05:
        return hdi_width * 2  # Penalize poor convergence
    
    return hdi_width

def run_tuning():
    # Project-root relative config path (from modeling/ subdir)
    config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    if not config_path.exists():
        logging.error(f"Config not found at {config_path}")
        return
    
    config = load_config()  # Load via updated func
    if config is None:
        logging.error("Failed to load config")
        return
    n_trials = config.get('n_trials', 10)  # Pull from YAML

    logging.info(f"Starting tuning with n_trials={n_trials}")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best priors:", study.best_params)
    print("Best HDI width:", study.best_value)
    
    # Update single config file with best params
    best_config = load_config()
    best_config.update(study.best_params)
    with open(config_path, "w") as f:
        yaml.dump({"model": best_config}, f)
    logging.info(f"Best config updated in {config_path}")

if __name__ == "__main__":
    run_tuning()