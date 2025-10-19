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
    Adds defaults: season_months=[5,10] for growing season.
    """
    config_path = Path("configs/data_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding='utf-8') as f:
        data_config = yaml.safe_load(f)["data"]
    # Normalize county names to uppercase for mapping
    data_config["counties_iowa_upper"] = {k.upper(): v for k, v in data_config["counties"]["iowa_fips"].items()}
    # Default growing season
    if "season_months" not in data_config:
        data_config["season_months"] = [5, 10]
    return data_config


def load_weather(config):
    """
    Load HRRR weather from monthly CSVs, aggregate growing-season vars per fips-year,
    add lagged versions (shift by 1 year per fips), filter to config fips_codes.
    Focus vars: precip (sum), avg/max temp (mean), VPD (mean). Handles missing files/NAs.
    """
    target_dir = Path(config["target_dir"])
    weather_dfs = []
    
    for year in config["focus_years"]:
        year_path = target_dir / "hrrr" / str(year) / "IA"
        if not year_path.exists():
            logging.warning(f"HRRR dir not found: {year_path}")
            continue
        
        monthly_dfs = []
        for month in range(1, 13):
            file_path = year_path / f"HRRR_19_IA_{year}-{month:02d}.csv"
            if not file_path.exists():
                logging.warning(f"HRRR file not found: {file_path}")
                continue
            
            try:
                df_month = pd.read_csv(file_path, encoding='utf-8')
                logging.info(f"Loaded {file_path}: shape {df_month.shape}")
                
                # Filter by state if present
                if 'State' in df_month.columns:
                    df_month = df_month[df_month['State'] == config["state"]]
                
                # Standardize fips (assume 'FIPS Code' is 5-digit str)
                df_month['fips'] = df_month['FIPS Code'].astype(str).str.zfill(5)
                df_month['year'] = int(year)
                df_month['month'] = month
                
                # Create date for season filter
                if 'Day' in df_month.columns and 'Month' in df_month.columns:
                    df_month['date'] = pd.to_datetime(df_month[['year', 'month', 'Day']])
                    season_mask = df_month['date'].dt.month.between(config["season_months"][0], config["season_months"][1])
                    df_season_month = df_month[season_mask]
                else:
                    # If no Day, assume monthly and filter months directly
                    month_mask = df_month['month'].between(config["season_months"][0], config["season_months"][1])
                    df_season_month = df_month[month_mask]
                
                if not df_season_month.empty:
                    monthly_dfs.append(df_season_month)
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        if monthly_dfs:
            df_year = pd.concat(monthly_dfs, ignore_index=True)
            
            # Aggregate per fips-year (handle NAs: fillna(0) post-agg)
            agg_dict = {
                'Avg Temperature (K)': 'mean',
                'Max Temperature (K)': 'mean',
                'Precipitation (kg m**-2)': 'sum',
                'Vapor Pressure Deficit (kPa)': 'mean'
            }
            df_agg = df_year.groupby(['fips', 'year']).agg(agg_dict).reset_index()
            df_agg.columns = ['fips', 'year', 'avg_temp_mean', 'max_temp_mean', 'precip_sum', 'vpd_mean']
            
            # Fill NAs
            df_agg[['avg_temp_mean', 'max_temp_mean', 'vpd_mean']] = df_agg[['avg_temp_mean', 'max_temp_mean', 'vpd_mean']].fillna(df_agg[['avg_temp_mean', 'max_temp_mean', 'vpd_mean']].mean())
            df_agg['precip_sum'] = df_agg['precip_sum'].fillna(0)
            
            # Filter to fips_codes
            if config["fips_codes"]:
                df_agg = df_agg[df_agg['fips'].isin(config["fips_codes"])]
            
            if not df_agg.empty:
                weather_dfs.append(df_agg)
                logging.info(f"Aggregated HRRR {year}: {len(df_agg)} rows")
    
    if weather_dfs:
        weather_df = pd.concat(weather_dfs, ignore_index=True)
        weather_df = weather_df.sort_values(['fips', 'year'])
        
        # Add lagged features (prior year per fips)
        for col in ['avg_temp_mean', 'max_temp_mean', 'precip_sum', 'vpd_mean']:
            weather_df[f'{col}_lag1'] = weather_df.groupby('fips')[col].shift(1)
            # Fill lag NAs with current or global mean (for 2017)
            weather_df[f'{col}_lag1'] = weather_df[f'{col}_lag1'].fillna(weather_df[col])
        
        processed_dir = Path(config["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        weather_df.to_csv(processed_dir / "weather_features.csv", index=False)
        logging.info(f"Weather features: {len(weather_df)} rows saved with lags")
        return weather_df
    else:
        logging.warning("No weather data loaded")
        return pd.DataFrame()


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


def merge_datasets(yields_df, ratio_df, weather_df, config):
    """
    Merge crop yields with cover crop ratios and weather (current + lagged) on fips and year.
    Aggregate yields across crops by mean per fips-year to combine all crops.
    Drop crop column.
    Produces processed dataset ready for modeling (yield ~ ratio + weather confounders).
    Optional: Generate profile report if config["profile_report"] is True.
    """
    if yields_df.empty or ratio_df.empty:
        logging.warning("Empty yields or ratio DataFrames; returning empty merged DF")
        return pd.DataFrame()
    
    if weather_df.empty:
        logging.warning("Empty weather DataFrame; merging without weather")
    
    # First aggregate yields across crops per fips-year
    aggregated_yields = yields_df.groupby(['fips', 'year'])['yield_bushels_acre'].mean().reset_index()
    aggregated_yields.rename(columns={'yield_bushels_acre': 'yield_bushels_acre_combined'}, inplace=True)
    
    # Merge with ratios
    merged_df = aggregated_yields.merge(ratio_df[['fips', 'year', 'cover_crop_ratio']], on=['fips', 'year'], how='inner')
    
    # Merge with weather (current + lags)
    if not weather_df.empty:
        weather_cols = [col for col in weather_df.columns if col not in ['fips', 'year']]
        merged_df = merged_df.merge(weather_df[['fips', 'year'] + weather_cols], on=['fips', 'year'], how='left')
        # Fill any new NAs with means
        num_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[num_cols] = merged_df[num_cols].fillna(merged_df[num_cols].mean())
    
    if config.get("profile_report", False):
        profile = ProfileReport(merged_df, title=f"{config['state']} Crops Profile", explorative=True, minimal=True)
        profile.to_file(Path(config["processed_dir"]) / f"{config['state'].lower()}_crops_profile.html")
        logging.info("Profile report generated")
    
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(processed_dir / f"{config['state'].lower()}_crops_processed.csv", index=False)
    
    logging.info(f"Merged dataset: {len(merged_df)} rows saved (aggregated across crops, with weather)")
    return merged_df


def create_time_series_splits(processed_df, config):
    """
    Create splits by randomly dividing counties (fips) across both years, 
    with ~80% in train and ~20% in test. This ensures train has most data from 
    both 2017 and 2022, while test has remaining parts from both years.
    Saves to splits_dir as CSV files for scalability.
    """
    if processed_df.empty:
        return
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Get unique fips across all data
    unique_fips = processed_df['fips'].unique()
    n_fips = len(unique_fips)
    n_train = int(0.8 * n_fips)
    
    # Randomly select train fips (without replacement)
    train_fips = np.random.choice(unique_fips, n_train, replace=False)
    test_fips = np.setdiff1d(unique_fips, train_fips)
    
    # Split dataframes based on fips
    train_df = processed_df[processed_df['fips'].isin(train_fips)].copy()
    test_df = processed_df[processed_df['fips'].isin(test_fips)].copy()
    
    splits_dir = Path(config["processed_dir"]) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(splits_dir / "train.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    logging.info(f"Splits created: Train {len(train_df)} rows ({train_df['year'].min()}-{train_df['year'].max()}) from {len(train_fips)} counties, "
                 f"Test {len(test_df)} rows ({test_df['year'].min()}-{test_df['year'].max()}) from {len(test_fips)} counties")


def run_pipeline():
    """
    Main pipeline orchestrator.
    Loads config, computes ratios, loads yields and weather, merges, creates splits.
    Handles messy data (NAs, filtering) and ensures reproducibility via config.
    Ready for scale-up: add modules for confounders (e.g., weather merge), hyperparam tuning, etc.
    """
    config = load_config()
    ratio_df = compute_cover_ratios(config)
    yields_df = load_crop_yields(config)
    weather_df = load_weather(config)
    processed_df = merge_datasets(yields_df, ratio_df, weather_df, config)
    create_time_series_splits(processed_df, config)
    
    if not processed_df.empty:
        print(f"Pipeline complete: {len(processed_df)} processed rows.")
        print(f"Unique fips: {processed_df['fips'].nunique()}")
    else:
        print("Pipeline warning: No data processed; check logs for issues.")


if __name__ == "__main__":
    run_pipeline()