import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from pathlib import Path
import yaml
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Silence deprecations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config():
    """
    Load configuration from YAML file.
    Expects keys: target_dir, processed_dir, state, focus_years, crops, fips_codes, counties (with 'iowa_fips' dict).
    """
    config_path = Path("configs/data_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding='utf-8') as f:
        data_config = yaml.safe_load(f)["data"]
    # Normalize county names to uppercase for mapping
    data_config["counties_iowa_upper"] = {k.upper(): v for k, v in data_config["counties"]["iowa_fips"].items()}
    return data_config


def compute_cover_ratios(config):
    """
    Load cover crop practice acres and farming land acres CSVs, compute ratios per county-year,
    map to FIPS codes, and save to cover_crop_ratio.csv.
    Handles NAs by setting ratio=0.0.
    """
    target_dir = Path(config["target_dir"])
    ate_dir = target_dir / "ate"
    cover_path = ate_dir / "cover_crop_practice_acres.csv"
    land_path = ate_dir / "farming_land_acres.csv"
    
    if not cover_path.exists() or not land_path.exists():
        raise FileNotFoundError(f"CSV files not found: {cover_path} or {land_path}")
    
    # Load and melt to long format
    cover_df = pd.read_csv(cover_path)
    land_df = pd.read_csv(land_path)
    
    cover_melt = pd.melt(
        cover_df, id_vars=['County'], value_vars=['Acres_2017', 'Acres_2022'],
        var_name='year_str', value_name='cover_acres'
    )
    cover_melt['year'] = cover_melt['year_str'].str.extract('(\d+)').astype(int)
    cover_melt['County'] = cover_melt['County'].str.upper()
    
    land_melt = pd.melt(
        land_df, id_vars=['County'], value_vars=['Acres_2017', 'Acres_2022'],
        var_name='year_str', value_name='land_acres'
    )
    land_melt['year'] = land_melt['year_str'].str.extract('(\d+)').astype(int)
    land_melt['County'] = land_melt['County'].str.upper()
    
    # Merge on County and year
    ratio_df = cover_melt.merge(land_melt, on=['County', 'year'], how='inner')
    
    # Compute ratio, handle division by zero or NA
    ratio_df['cover_crop_ratio'] = np.where(
        (ratio_df['cover_acres'].isna()) | (ratio_df['land_acres'] == 0),
        0.0,
        ratio_df['cover_acres'] / ratio_df['land_acres']
    )
    ratio_df['cover_crop_ratio'] = ratio_df['cover_crop_ratio'].fillna(0.0)
    
    # Map to FIPS
    ratio_df['fips'] = ratio_df['County'].map(config["counties_iowa_upper"])
    ratio_df = ratio_df.dropna(subset=['fips'])  # Drop unmapped counties
    ratio_df['fips'] = ratio_df['fips'].astype(str)
    
    # Filter to config fips_codes if subset for testing
    if config["fips_codes"]:
        ratio_df = ratio_df[ratio_df['fips'].isin(config["fips_codes"])]
    
    # Select and save
    ratio_out = ratio_df[['fips', 'year', 'cover_crop_ratio']].sort_values(['fips', 'year'])
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    ratio_out.to_csv(processed_dir / "cover_crop_ratio.csv", index=False)
    
    logging.info(f"Computed ratios: {len(ratio_df)} rows saved to {processed_dir / 'cover_crop_ratio.csv'}")
    return ratio_df


def load_crop_yields(config):
    """
    Load USDA crop yield data for specified crops, years, and state.
    Produces a single DataFrame with fips, year, crop, yield_bushels_acre.
    Filters to config fips_codes.
    """
    target_dir = Path(config["target_dir"])
    yield_dfs = []
    
    for year in config["focus_years"]:
        for crop in config["crops"]:
            usda_path = target_dir / "usda" / str(year) / crop / f"USDA_{crop}_County_{year}.csv"
            if not usda_path.exists():
                logging.warning(f"USDA file not found: {usda_path}")
                continue
            
            try:
                df_year = pd.read_csv(usda_path, encoding='utf-8')
                logging.info(f"Loaded {usda_path}: shape {df_year.shape}")
                
                # Filter by state
                df_year = df_year[df_year['state_name'] == config["state"]]
                if df_year.empty:
                    logging.info(f"No data for {crop} in {config['state']} {year}")
                    continue
                
                # Extract key columns
                df_year['year'] = int(year)
                df_year['crop'] = crop
                df_year['fips'] = '19' + df_year['county_ansi'].astype(str).str.zfill(3)
                df_year['yield_bushels_acre'] = pd.to_numeric(df_year['YIELD, MEASURED IN BU / ACRE'], errors='coerce')
                
                # Filter to fips_codes subset
                if config["fips_codes"]:
                    df_year = df_year[df_year['fips'].isin(config["fips_codes"])]
                
                if not df_year.empty:
                    yield_dfs.append(df_year[['fips', 'year', 'crop', 'yield_bushels_acre']])
                
            except Exception as e:
                logging.error(f"Error processing {usda_path}: {e}")
    
    if yield_dfs:
        yields_df = pd.concat(yield_dfs, ignore_index=True)
        # Drop rows with NaN yields
        yields_df = yields_df.dropna(subset=['yield_bushels_acre'])
        logging.info(f"Loaded crop yields: {len(yields_df)} rows across {len(yields_df['year'].unique())} years and {len(yields_df['crop'].unique())} crops")
        return yields_df
    else:
        logging.warning("No yield data loaded")
        return pd.DataFrame()


def merge_datasets(yields_df, ratio_df, config):
    """
    Merge crop yields with cover crop ratios on fips and year.
    Aggregate yields across crops by mean per fips-year to combine all crops.
    Drop crop column.
    Produces processed dataset ready for modeling (yield ~ ratio + future confounders).
    Optional: Generate profile report if config["profile_report"] is True.
    """
    if yields_df.empty or ratio_df.empty:
        logging.warning("Empty input DataFrames; returning empty merged DF")
        return pd.DataFrame()
    
    # First aggregate yields across crops per fips-year
    aggregated_yields = yields_df.groupby(['fips', 'year'])['yield_bushels_acre'].mean().reset_index()
    aggregated_yields.rename(columns={'yield_bushels_acre': 'yield_bushels_acre_combined'}, inplace=True)
    
    # Merge with ratios
    merged_df = aggregated_yields.merge(ratio_df[['fips', 'year', 'cover_crop_ratio']], on=['fips', 'year'], how='inner')
    
    if config.get("profile_report", False):
        profile = ProfileReport(merged_df, title=f"{config['state']} Crops Profile", explorative=True, minimal=True)
        profile.to_file(Path(config["processed_dir"]) / f"{config['state'].lower()}_crops_profile.html")
        logging.info("Profile report generated")
    
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(processed_dir / f"{config['state'].lower()}_crops_processed.csv", index=False)
    
    logging.info(f"Merged dataset: {len(merged_df)} rows saved (aggregated across crops)")
    return merged_df


def create_time_series_splits(processed_df, config):
    """
    Create walk-forward splits for time series data.
    With focus_years=["2017","2022"], uses 2017 as train, 2022 as test (no validation due to limited years).
    Saves to splits_dir as CSV files for scalability.
    Modular for future years: extend by sorting years and expanding windows.
    """
    if processed_df.empty:
        return
    
    processed_df['year'] = pd.to_numeric(processed_df['year'])
    years = sorted(processed_df['year'].unique())
    
    # For current focus_years, simple split; scalable to walk-forward for more years
    train_df = processed_df[processed_df['year'] == 2017]
    test_df = processed_df[processed_df['year'] == 2022]
    
    splits_dir = Path(config["processed_dir"]) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(splits_dir / "train.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    logging.info(f"Splits created: Train {len(train_df)} rows ({train_df['year'].min()}-{train_df['year'].max()}), "
                 f"Test {len(test_df)} rows ({test_df['year'].min()}-{test_df['year'].max()})")
    
    # For future scalability: if more years, implement rolling windows
    # e.g., for i in range(len(years)-2): train = years[:i+1], val=years[i+1], test=years[i+2]


def run_pipeline():
    """
    Main pipeline orchestrator.
    Loads config, computes ratios, loads yields, merges, creates splits.
    Handles messy data (NAs, filtering) and ensures reproducibility via config.
    Ready for scale-up: add modules for confounders (e.g., weather merge), hyperparam tuning, etc.
    """
    config = load_config()
    ratio_df = compute_cover_ratios(config)
    yields_df = load_crop_yields(config)
    processed_df = merge_datasets(yields_df, ratio_df, config)
    create_time_series_splits(processed_df, config)
    
    if not processed_df.empty:
        print(f"Pipeline complete: {len(processed_df)} processed rows.")
        print(f"Unique fips: {processed_df['fips'].nunique()}")
    else:
        print("Pipeline warning: No data processed; check logs for issues.")


if __name__ == "__main__":
    run_pipeline()