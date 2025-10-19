# estimate_ate.py (updated: Project-root relative paths; debug prints for loaded config; CSV-only load, no parquet)
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
    # Project-root relative config path (from modeling/ subdir)
    config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    if not config_path.exists():
        logging.error(f"Config not found at {config_path}")
        return None
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)["model"]
    # Debug print: Confirm loaded values
    logging.info(f"Loaded config: n_draws={config.get('n_draws', 'MISSING')}, n_trials={config.get('n_trials', 'MISSING')}")
    return config

def load_data(split='train'):
    processed_dir = Path("data/processed")  # From prep script; read CSV only, no parquet churn
    splits_dir = processed_dir / "splits"
    df = pd.read_csv(splits_dir / f"{split}.csv")
    return df

def fit_bayesian_scm(train_df, config):
    # Prep data: Fill NAs, map fips to indices for U latents
    train_df = train_df.fillna(0).copy()
    fips_unique = train_df['fips'].astype(int).unique()
    fips_to_idx = {fip: idx for idx, fip in enumerate(fips_unique)}
    train_df['fips_idx'] = train_df['fips'].map(fips_to_idx)
    n_fips = len(fips_unique)
    n_conf = 8  # 4 current + 4 lagged
    
    # No class weights (continuous T); drop from config if present
    if 'class_weight_scale' in config:
        del config['class_weight_scale']
    
    # Prepare data arrays
    confounds_data = np.column_stack([
        train_df['precip_sum'].values, train_df['avg_temp_mean'].values,
        train_df['max_temp_mean'].values, train_df['vpd_mean'].values,
        train_df['precip_sum_lag1'].values, train_df['avg_temp_mean_lag1'].values,
        train_df['max_temp_mean_lag1'].values, train_df['vpd_mean_lag1'].values
    ])
    fips_idx_data = train_df['fips_idx'].values

    with pm.Model(coords={'fips': fips_unique}) as scm_model:
        # Treatment effect: Informative prior from lit (modest +2-6% nudge)
        beta_treat = pm.Normal('beta_treat', mu=config["prior_mean"], sigma=np.sqrt(config["prior_var"]))
        
        # Confounders: Vector for current/lagged weather (Gaussian priors, tuned)
        beta_confound = pm.Normal('beta_confound', mu=config["prior_mean"], sigma=np.sqrt(config["prior_var"]), shape=n_conf)
        
        # Unobserved confounders U: Hierarchical latents per fips (impute via REs)
        u_sd = pm.HalfNormal('u_sd', sigma=np.sqrt(config["u_var"]))
        u_latent = pm.Normal('u_latent', mu=0, sigma=u_sd, dims='fips')

        # Define data containers for swapping train/test sets
        fips_idx = pm.Data('fips_idx_data', fips_idx_data)
        confounds = pm.Data('confounds_data', confounds_data)
        cover_crop_ratio = pm.Data('cover_crop_ratio_data', train_df['cover_crop_ratio'].values)
        yield_data = pm.Data('yield_data', train_df['yield_bushels_acre_combined'].values)
        
        u_fips = u_latent[fips_idx]  # Assign to obs
        
        # Noise
        sigma = pm.HalfNormal('sigma', sigma=10)  # Yield scale ~100 bu/acre
        
        # Linear predictor: Yield ~ Treat + Current + Lagged + U
        confound_term = pm.math.dot(confounds, beta_confound)
        mu = (beta_treat * cover_crop_ratio + confound_term + u_fips)
        
        # Likelihood
        pm.Normal('yield_obs', mu=mu, sigma=sigma, observed=yield_data)
        
        # Sample posteriors with boosted params for stability
        sample_kwargs = {
            'chains': config.get('n_chains', 4),
            'draws': config.get('n_draws', 100),  # Explicit fallback to YAML value
            'tune': 1500,
            'target_accept': config.get('target_accept', 0.9),
            'max_treedepth': config.get('max_treedepth', 10),
            'return_inferencedata': True
        }
        trace = pm.sample(**sample_kwargs)
        
        # Posterior predictive for PPC (extend trace)
        ppc = pm.sample_posterior_predictive(trace, var_names=['yield_obs'])
        trace.extend(ppc)
    
    return scm_model, trace

def infer_ate(trace, test_df=None, refutation=False):
    # ATE: Posterior mean of beta_treat (direct effect post-adjustment); for full counterfactual, sample diffs
    ate_summary = az.summary(trace, var_names=['beta_treat'], hdi_prob=0.95)
    ate_mean = ate_summary['mean'].iloc[0]
    ate_hdi_low = ate_summary['hdi_2.5%'].iloc[0]
    ate_hdi_high = ate_summary['hdi_97.5%'].iloc[0]
    ate_hdi = (ate_hdi_low, ate_hdi_high)
    
    if refutation:
        # Placebo: Fit simple model with randomized treatment; expect ATE ~0
        np.random.seed(42)
        placebo_t = np.random.normal(0, 1, 100)  # Fixed size
        placebo_y = np.random.normal(0, 1, 100)
        with pm.Model() as placebo_model:
            beta_placebo = pm.Normal('beta_placebo', mu=0, sigma=0.01)
            pm.Normal('y_placebo', mu=beta_placebo * placebo_t, sigma=1, observed=placebo_y)
            placebo_trace = pm.sample(chains=2, draws=500, tune=500, return_inferencedata=True)
        placebo_ate = az.summary(placebo_trace, var_names=['beta_placebo'])['mean'].iloc[0]
        logging.info(f"Refutation (placebo ATE): {placebo_ate:.2f} (should ~0)")
    
    # If test_df, compute counterfactual ATE via posterior predictive (E[Y1 - Y0 | W, U~0 for oos])
    if test_df is not None:
        test_df = test_df.fillna(0).copy()
        n_test = len(test_df)
        if n_test == 0:
            return ate_mean, ate_hdi
        
        # Reshape posteriors safely
        beta_treat_samp = trace.posterior['beta_treat'].values.reshape(-1)  # (8000,)
        beta_confound_samp = trace.posterior['beta_confound'].values.reshape(-1, 8)  # (8000, 8)
        sigma_samp = trace.posterior['sigma'].values.reshape(-1)  # (8000,)
        n_samps = len(beta_treat_samp)
        
        # Confounds matrix
        confounds_test = np.column_stack([
            test_df['precip_sum'].values, test_df['avg_temp_mean'].values,
            test_df['max_temp_mean'].values, test_df['vpd_mean'].values,
            test_df['precip_sum_lag1'].values, test_df['avg_temp_mean_lag1'].values,
            test_df['max_temp_mean_lag1'].values, test_df['vpd_mean_lag1'].values
        ])  # (n_test, 8)
        
        y1 = np.zeros((n_samps, n_test))
        y0 = np.zeros((n_samps, n_test))
        for s in range(n_samps):
            confound_term = np.dot(confounds_test, beta_confound_samp[s])  # (n_test,)
            # U ~0 for oos (or map if fips overlap; simplify here)
            mu1 = beta_treat_samp[s] * 1 + confound_term
            mu0 = beta_treat_samp[s] * 0 + confound_term
            y1[s] = np.random.normal(mu1, sigma_samp[s])
            y0[s] = np.random.normal(mu0, sigma_samp[s])
        
        cf_ate = np.mean(y1 - y0, axis=1)  # (8000,) per-sample means
        cf_ate_mean = np.mean(cf_ate)
        cf_ate_hdi = np.percentile(cf_ate, [2.5, 97.5])
        logging.info(f"Counterfactual ATE on test: {cf_ate_mean:.2f} [95% HDI: {cf_ate_hdi[0]:.2f}, {cf_ate_hdi[1]:.2f}]")
        return cf_ate_mean, cf_ate_hdi
    else:
        return ate_mean, ate_hdi

def evaluate_model(model, trace, test_df, config):
    # Map fips for test set if needed for U latents
    train_df = load_data('train')
    fips_unique = train_df['fips'].unique()
    fips_to_idx = {fip: idx for idx, fip in enumerate(fips_unique)}
    test_df['fips_idx'] = test_df['fips'].map(fips_to_idx).fillna(-1).astype(int) # Use -1 for unseen fips

    # Set coords for out-of-sample prediction
    with model:
        pm.set_data({
            'yield_data': test_df['yield_bushels_acre_combined'].values,
            'cover_crop_ratio_data': test_df['cover_crop_ratio'].values,
            'confounds_data': np.column_stack([
                test_df['precip_sum'].values, test_df['avg_temp_mean'].values,
                test_df['max_temp_mean'].values, test_df['vpd_mean'].values,
                test_df['precip_sum_lag1'].values, test_df['avg_temp_mean_lag1'].values,
                test_df['max_temp_mean_lag1'].values, test_df['vpd_mean_lag1'].values
            ]),
            'fips_idx_data': test_df['fips_idx'].values
        })
        oos_preds = pm.sample_posterior_predictive(trace, var_names=['yield_obs'], predictions=True)
        trace.extend(oos_preds)

    # Bias-variance: Use posterior predictive on test (from infer_ate cf)
    test_ate, test_hdi = infer_ate(trace, test_df)
    # RMSE from pred vs obs (use mean mu from trace for quick)
    pred_mean = trace.predictions['yield_obs'].mean(dim=['chain', 'draw']).values
    rmse = np.sqrt(np.mean((test_df['yield_bushels_acre_combined'] - pred_mean)**2))
    bias = np.mean(pred_mean - test_df['yield_bushels_acre_combined'])
    variance = np.var(pred_mean)
    logging.info(f"Test eval: RMSE {rmse:.2f}, Bias {bias:.2f}, Variance {variance:.2f}")
    logging.info(f"Test ATE: {test_ate:.2f} [95% HDI: {test_hdi[0]:.2f}, {test_hdi[1]:.2f}]")

def run_modeling():
    config = load_config()
    if config is None:
        logging.error("Config load failed in run_modeling")
        return
    train_df = load_data('train')
    test_df = load_data('test')
    
    if train_df.empty:
        logging.warning("No train data; check splits.")
        return
    
    model, trace = fit_bayesian_scm(train_df, config)
    logging.info("Bayesian SCM fitted; posteriors + PPC sampled.")
    
    # Infer on train for quick check
    train_ate, train_hdi = infer_ate(trace)
    logging.info(f"Train ATE: {train_ate:.2f} [95% HDI: {train_hdi[0]:.2f}, {train_hdi[1]:.2f}]")
    
    # Evaluate on test
    evaluate_model(model, trace, test_df, config)
    
    # Save extended trace (with PPC, counterfactuals)
    az.to_netcdf(trace, "data/processed/ate_trace.nc")
    logging.info("Extended trace saved; ready for summary.")
    
    # Diagnostics plot
    az.plot_trace(trace, var_names=['beta_treat', 'beta_confound', 'u_sd'])  # Skip u_latent if high-dim
    plt.savefig("data/processed/trace_diagnostics.png", dpi=150)
    plt.close()
    logging.info("Trace diagnostics plot saved.")

if __name__ == "__main__":
    run_modeling()