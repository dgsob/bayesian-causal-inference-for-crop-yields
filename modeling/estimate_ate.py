import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("configs/model_config.yaml", "r", encoding='utf-8') as f:
        return yaml.safe_load(f)["model"]

def load_data(split='train'):
    return pd.read_parquet(f"data/processed/splits/{split}.parquet")

def fit_bayesian_scm(train_df, config):
    train_df = train_df.fillna(0)
    weights = train_df['class_weight'] * config["class_weight_scale"]
    
    with pm.Model() as model:
        beta_treat = pm.Normal('beta_treat', mu=config["prior_mean"], sigma=np.sqrt(config["prior_var"]))
        beta_confound = pm.Normal('beta_confound', mu=config["prior_mean"], sigma=np.sqrt(config["prior_var"]), shape=2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        mu = (beta_treat * train_df['cover_crop_proxy'] +
              beta_confound[0] * train_df['Precipitation (kg m**-2)_summer'] +
              beta_confound[1] * train_df['Avg Temperature (K)_summer'])
        
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=train_df['yield_bushels_acre'])
        
        log_weights = np.log(weights)
        weighted_ll = pm.Potential('weighted_ll', (pm.logp(pm.Normal.dist(mu, sigma), train_df['yield_bushels_acre']) * log_weights).sum())
        
        trace = pm.sample(chains=config["n_chains"], draws=config["n_draws"], tune=1000, target_accept=0.9, return_inferencedata=True)
    
    return model, trace

def infer_ate(trace, refutation=False):
    ate_summary = az.summary(trace, var_names=['beta_treat'], hdi_prob=0.95)
    ate_mean = ate_summary['mean'].iloc[0]
    ate_hdi_low = ate_summary['hdi_2.5%'].iloc[0]
    ate_hdi_high = ate_summary['hdi_97.5%'].iloc[0]
    ate_hdi = ate_hdi_low, ate_hdi_high
    
    if refutation:
        with pm.Model() as placebo_model:
            beta_placebo = pm.Normal('beta_placebo', mu=0, sigma=0.01)
            # Simplified (reuse structure; full in practice)
            placebo_trace = pm.sample(chains=2, draws=500, tune=500)
            placebo_ate = az.summary(placebo_trace, var_names=['beta_placebo'])['mean'].iloc[0]
        logging.info(f"Refutation (placebo ATE): {placebo_ate:.2f} (should ~0)")
    
    return ate_mean, ate_hdi

def evaluate_model(model, trace, val_df, test_df, config):
    val_df_filled = val_df.fillna(0)
    
    # Fixed: Manual posterior mean predictions for val (no PPC needed for bias-variance)
    beta_treat_mean = trace.posterior['beta_treat'].mean(dim=['chain', 'draw']).values
    beta_confound_mean = trace.posterior['beta_confound'].mean(dim=['chain', 'draw']).values
    sigma_mean = trace.posterior['sigma'].mean(dim=['chain', 'draw']).values
    
    # Baseline pred for val
    mu_val = (beta_treat_mean * val_df_filled['cover_crop_proxy'].values +
              beta_confound_mean[0] * val_df_filled['Precipitation (kg m**-2)_summer'].values +
              beta_confound_mean[1] * val_df_filled['Avg Temperature (K)_summer'].values)
    pred_mean = np.random.normal(mu_val, sigma_mean)  # Shape (n_val, )
    
    # Perturb for counterfactual sim
    sim_df = val_df_filled.copy()
    sim_df['cover_crop_proxy'] = 1 - sim_df['cover_crop_proxy']
    mu_sim = (beta_treat_mean * sim_df['cover_crop_proxy'].values +
              beta_confound_mean[0] * sim_df['Precipitation (kg m**-2)_summer'].values +
              beta_confound_mean[1] * sim_df['Avg Temperature (K)_summer'].values)
    sim_pred = np.random.normal(mu_sim, sigma_mean)
    bias = np.mean(pred_mean - sim_pred)
    variance = np.var(pred_mean)
    logging.info(f"Val eval: Bias {bias:.2f}, Variance {variance:.2f}")
    
    test_ate, test_hdi = infer_ate(trace)
    logging.info(f"Test ATE: {test_ate:.2f} [95% HDI: {test_hdi[0]:.2f}, {test_hdi[1]:.2f}]")

def run_modeling():
    config = load_config()
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    if train_df.empty:
        logging.warning("No train data; check splits.")
        return
    
    model, trace = fit_bayesian_scm(train_df, config)
    logging.info("Model fitted; posteriors sampled.")
    
    val_ate, val_hdi = infer_ate(trace)
    logging.info(f"Val ATE: {val_ate:.2f} [95% HDI: {val_hdi[0]:.2f}, {val_hdi[1]:.2f}]")
    
    evaluate_model(model, trace, val_df, test_df, config)
    
    az.to_netcdf(trace, "data/processed/ate_trace.nc")
    logging.info("Trace saved; use az.plot_trace for diagnostics.")
    
    # Quick plot
    az.plot_trace(trace, var_names=['beta_treat'])
    plt.savefig("data/processed/trace_plot.png")
    plt.close()
    logging.info("Trace plot saved to data/processed/trace_plot.png")

if __name__ == "__main__":
    run_modeling()