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
    with open("configs/data_config.yaml", "r", encoding='utf-8') as f:
        return yaml.safe_load(f)["data"]

def load_usda_data(config):
    usda_dfs = []
    target_dir = Path(config["target_dir"])
    for year in config["years"]:
        usda_path = target_dir / f"usda/{year}/Corn/USDA_{config['crop_type']}_County_{year}.csv"
        if usda_path.exists():
            try:
                df_year = pd.read_csv(usda_path, encoding='utf-8')
                logging.info(f"Loaded {usda_path}: columns {list(df_year.columns)}")
                df_year['fips'] = '19' + df_year['county_ansi'].astype(str).str.zfill(3)
                df_year['yield_bushels_acre'] = pd.to_numeric(df_year['YIELD, MEASURED IN BU / ACRE'], errors='coerce')
                df_year['Planted_acres'] = np.nan
                df_year['year'] = int(year)
                df_year = df_year[df_year['fips'].isin(config["fips_codes"])]
                if not df_year.empty:
                    usda_dfs.append(df_year[['fips', 'year', 'yield_bushels_acre', 'Planted_acres']])
            except Exception as e:
                logging.error(f"Error processing {usda_path}: {e}")
    usda_df = pd.concat(usda_dfs, ignore_index=True) if usda_dfs else pd.DataFrame()
    if not usda_df.empty:
        logging.info(f"Loaded USDA: {len(usda_df)} rows across {len(usda_df['year'].unique())} years")
    return usda_df

def load_hrrr_data(config):
    weather_dfs = []
    target_dir = Path(config["target_dir"])
    for year in config["years"]:
        year_dfs = []
        for month in range(1, 13):
            hrrr_path = target_dir / f"hrrr/{year}/IA/HRRR_19_IA_{year}-{month:02d}.csv"
            if hrrr_path.exists():
                try:
                    df_month = pd.read_csv(hrrr_path, encoding='utf-8')
                    df_month['date'] = pd.to_datetime(df_month[['Year', 'Month', 'Day']])
                    df_month['season'] = pd.cut(df_month['date'].dt.month, bins=[0,3,6,9,12], labels=['winter','spring','summer','fall'])
                    df_month['fips'] = df_month['FIPS Code'].astype(str).str.zfill(5)
                    df_month = df_month[df_month['fips'].isin(config["fips_codes"])]
                    seasonal_summary = df_month.groupby(['fips', 'season'], observed=True).agg({
                        'Avg Temperature (K)': 'mean',
                        'Precipitation (kg m**-2)': 'sum'
                    }).reset_index()
                    seasonal_summary['year'] = int(year)
                    year_dfs.append(seasonal_summary)
                except Exception as e:
                    logging.error(f"Error processing {hrrr_path}: {e}")
        if year_dfs:
            year_weather = pd.concat(year_dfs, ignore_index=True)
            year_weather_pivot = year_weather.pivot_table(index=['fips', 'year'], columns='season', values=['Avg Temperature (K)', 'Precipitation (kg m**-2)'], aggfunc='first', observed=True).reset_index()
            year_weather_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in year_weather_pivot.columns]
            weather_dfs.append(year_weather_pivot)
    weather_df = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()
    if not weather_df.empty:
        logging.info(f"Loaded HRRR: {len(weather_df)} rows")
    return weather_df

def merge_data(usda_df, weather_df):
    if not usda_df.empty and not weather_df.empty:
        return usda_df.merge(weather_df, on=['fips', 'year'], how='outer')
    return usda_df if not usda_df.empty else weather_df if not weather_df.empty else pd.DataFrame()

def engineer_features(merged_df, config):
    if merged_df.empty:
        return merged_df
    merged_df['irrigation_proxy'] = (merged_df.get('Precipitation (kg m**-2)_summer', np.nan) < config["irrigation_proxy_threshold"]).astype(int)
    np.random.seed(42)
    merged_df['cover_crop_proxy'] = np.random.binomial(1, 0.2, len(merged_df))
    class_counts = merged_df['cover_crop_proxy'].value_counts()
    total = class_counts.sum()
    merged_df['class_weight'] = merged_df['cover_crop_proxy'].map({k: total / (len(class_counts) * class_counts[k]) for k in class_counts.index})
    if config["profile_report"]:
        profile = ProfileReport(merged_df, title="Iowa Corn Profile", explorative=True, minimal=True)
        profile.to_file(Path(config["processed_dir"]) / "ia_corn_profile.html")
    return merged_df

def run_pipeline():
    config = load_config()
    usda_df = load_usda_data(config)
    weather_df = load_hrrr_data(config)
    merged_df = merge_data(usda_df, weather_df)
    processed_df = engineer_features(merged_df, config)

    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not processed_df.empty:
        processed_df.to_parquet(processed_dir / "ia_corn_initial.parquet", index=False)
        print(f"Processed: {len(processed_df)} rows saved.")
        print(f"Imbalance weights: {processed_df['class_weight'].value_counts(normalize=True)}")
        
        # Walk-forward splits
        processed_df['year'] = pd.to_numeric(processed_df['year'])
        splits_dir = processed_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        train_df = processed_df[processed_df['year'] <= 2020]
        val_df = processed_df[processed_df['year'] == 2021]
        test_df = processed_df[processed_df['year'] == 2022]
        train_df.to_parquet(splits_dir / "train.parquet", index=False)
        val_df.to_parquet(splits_dir / "val.parquet", index=False)
        test_df.to_parquet(splits_dir / "test.parquet", index=False)
        print(f"Splits: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    else:
        print("No data; check logs.")

if __name__ == "__main__":
    run_pipeline()