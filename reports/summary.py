import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import logging
import numpy as np  # For sqrt in title

def load_trace():
    config = yaml.safe_load(open("configs/model_config.yaml"))["model"]
    trace = az.from_netcdf("data/processed/ate_trace.nc")  # Refit with best priors
    return trace, config

def generate_summary(trace, config):
    Path("reports").mkdir(exist_ok=True)
    
    # Posterior density plot for ATE using plot_posterior: Full distribution with mean, HDI
    ate_summary = az.summary(trace, var_names=['beta_treat'], hdi_prob=0.95)
    post_mean = ate_summary['mean'].iloc[0]
    hdi_low, hdi_high = ate_summary['hdi_2.5%'].iloc[0], ate_summary['hdi_97.5%'].iloc[0]
    
    ax = az.plot_posterior(trace, var_names=['beta_treat'], point_estimate='mean', hdi_prob=0.95, kind='kde')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No Effect (0)')  # Null line
    ax.set_title(f"Bayesian ATE Posterior for Treatment\n"
                 f"Mean: {post_mean:.3f} | 95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]\n"
                 f"Priors: N({config['prior_mean']:.3f}, {np.sqrt(config['prior_var']):.3f})",
                 fontsize=12)
    ax.set_xlabel("Treatment Effect (bushels/acre)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/ate_posterior_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Posterior predictive vs. observed (load train for goodness-of-fit check, as PPC validates model replication of fitting data)
    if 'posterior_predictive' in trace:
        train_df = pd.read_parquet("data/processed/splits/train.parquet")
        az.plot_ppc(trace, data_pairs={'y_obs': train_df['yield_bushels_acre']})
        plt.savefig("reports/ppc_train.png")
        plt.close()
        logging.info("PPC plot generated using train data for model validation")
    else:
        logging.warning("Trace missing 'posterior_predictive' group; skipping PPC. Ensure modeling script includes idata.extend(pm.sample_posterior_predictive(idata)) before saving netcdf.")
    
    # For test set, generate simple prediction summary (e.g., mean posterior predictive if available, or refit model for out-of-sample)
    # Placeholder: load test and compute naive mean for now; extend with full posterior predictive on test in modeling module
    test_df = pd.read_parquet("data/processed/splits/test.parquet")
    if 'posterior_predictive' in trace and 'y_pred' in trace.posterior_predictive:
        pred_mean = trace.posterior_predictive['y_pred'].mean(dim=['chain', 'draw']).values
        test_df['pred_mean'] = pred_mean[:len(test_df)]  # Align shapes assuming sequential indexing
        test_df.to_parquet("data/processed/splits/test_with_preds.parquet", index=False)
        rmse = ((test_df['yield_bushels_acre'] - test_df['pred_mean'])**2).mean()**0.5
        logging.info(f"Test RMSE: {rmse:.2f} (extend modeling for full test PPC if needed)")
    else:
        logging.info("No posterior_predictive for test viz; consider walk-forward validation in modeling module")
    
    # Summary table
    summary = az.summary(trace, hdi_prob=0.95)
    summary.to_csv("reports/model_summary.csv")
    print(summary[['mean', 'hdi_2.5%', 'hdi_97.5%']])  # Print key stats
    
    logging.info("Summary generated: reports/ate_posterior_density.png, ppc_train.png (if available), model_summary.csv")

if __name__ == "__main__":
    trace, config = load_trace()
    generate_summary(trace, config)