import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import logging

def load_trace():
    config = yaml.safe_load(open("configs/model_config.yaml"))["model"]
    trace = az.from_netcdf("data/processed/ate_trace.nc")  # Refit with best priors
    return trace, config

def generate_summary(trace, config):
    Path("reports").mkdir(exist_ok=True)
    
    # ATE forest plot
    az.plot_forest(trace, var_names=['beta_treat'], combined=True)
    plt.title(f"ATE with Best Priors (mean={config['prior_mean']}, var={config['prior_var']})")
    plt.savefig("reports/ate_forest.png")
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
    
    logging.info("Summary generated: reports/ate_forest.png, ppc_train.png (if available), model_summary.csv")

if __name__ == "__main__":
    trace, config = load_trace()
    generate_summary(trace, config)