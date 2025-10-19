# summary.py
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trace():
    config_path = Path("configs/model_config.yaml")
    config = yaml.safe_load(open(config_path))["model"]
    trace = az.from_netcdf("data/processed/ate_trace.nc")
    return trace, config

def generate_summary(trace, config):
    reports_dir = Path("results")
    reports_dir.mkdir(exist_ok=True)
    
    # ATE Posterior: Density with HDI, null line
    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_posterior(trace, var_names=['beta_treat'], point_estimate='mean', hdi_prob=0.95, kind='kde', ax=ax)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No Effect (0)')
    post_mean = trace.posterior['beta_treat'].mean().values
    hdi = az.hdi(trace.posterior['beta_treat'], hdi_prob=0.95).beta_treat.values
    ax.set_title(f"Bayesian ATE Posterior (Adjusted for Weather + U)\n"
                 f"Mean: {post_mean:.3f} | 95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]\n"
                 f"Priors: Treat N({config['prior_mean']:.3f}, {np.sqrt(config['prior_var']):.3f}), "
                 f"U Var: {config['u_var']:.3f}",
                 fontsize=12)
    ax.set_xlabel("Treatment Effect (bushels/acre)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "ate_posterior_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # PPC: Train vs. posterior predictive (goodness-of-fit)
    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_ppc(trace, ax=ax)
    ax.set_title("Posterior Predictive Check (Train)")
    plt.savefig(reports_dir / "ppc_train.png", dpi=150)
    plt.close()
    logging.info("PPC plot generated for train validation")
    
    # Test Predictions: RMSE summary (load enhanced test if available)
    test_df = pd.read_csv("data/processed/splits/test.csv")
    if 'pred_mean' in test_df.columns:  # From eval if added
        rmse = np.sqrt(np.mean((test_df['yield_bushels_acre_combined'] - test_df['pred_mean'])**2))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(test_df['yield_bushels_acre_combined'], test_df['pred_mean'], alpha=0.6)
        ax.plot([0, max(test_df['yield_bushels_acre_combined'])], [0, max(test_df['yield_bushels_acre_combined'])], 'r--')
        ax.set_xlabel('Observed Yields'); ax.set_ylabel('Predicted Yields')
        ax.set_title(f"Test Predictions (RMSE: {rmse:.2f})")
        plt.savefig(reports_dir / "test_predictions.png", dpi=150)
        plt.close()
        logging.info(f"Test predictions plot saved; RMSE: {rmse:.2f}")
    
    # U Latents: Posterior for unobserved confounders (e.g., soil proxies)
    # Manually create a ridge plot to avoid dtype issues in az.plot_posterior
    u_latent_data = trace.posterior["u_latent"]
    fips_coords = u_latent_data.coords["fips"].values
    
    fig, axes = plt.subplots(
        len(fips_coords), 1, figsize=(8, 12), sharex=True
    )
    
    for i, fips in enumerate(fips_coords):
        ax = axes[i]
        # Flatten the (chain, draw) dimensions into a 1D numpy array for plotting
        kde_values = u_latent_data.sel(fips=fips).values.flatten()
        az.plot_kde(kde_values, ax=ax)
        ax.set_yticks([])
        ax.set_ylabel(fips, rotation=0, ha='right', va='center')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel("u_latent value")
    ax.set_title("Posteriors for Unobserved Confounders (U per FIPS)")
    plt.savefig(reports_dir / "u_latent_posteriors.png", dpi=150)
    plt.close()
    
    # Full Summary Table: Key params (ATE, confounds, U sd, sigma)
    summary = az.summary(trace, hdi_prob=0.95, var_names=['beta_treat', 'beta_confound', 'u_sd', 'sigma'])
    summary.to_csv(reports_dir / "model_summary.csv")
    print(summary[['mean', 'hdi_2.5%', 'hdi_97.5%']])  # Print stats
    
    # Bias-Variance Decomp Plot (simple bar from eval logs; extend with sims)
    # Placeholder: Assume eval logged bias/var; plot if saved
    logging.info("Summary generated: Check reports/ for plots, model_summary.csv")
    logging.info("For refutation/E-value: Run infer_ate with refutation=True in modeling.")

if __name__ == "__main__":
    trace, config = load_trace()
    generate_summary(trace, config)