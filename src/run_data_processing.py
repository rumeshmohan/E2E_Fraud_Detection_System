import pandas as pd
import os
import sys

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import your module
from src.feature_engineering import create_all_features

# --- Define File Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Fraud_Data.csv')
CURATED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'curated_dataset.parquet')

def main():
    """
    Main function to run the data processing pipeline.
    1. Load raw data
    2. Create features
    3. Save curated data
    """
    print("Starting data processing pipeline...")

    # 1. Load Raw Data
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        print(f"Raw data loaded from {RAW_DATA_PATH}. Shape: {df_raw.shape}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    # 2. Create Features
    # This calls the function you already wrote and tested!
    print("Creating features...")
    df_features = create_all_features(df_raw)
    print("Feature engineering complete.")

    # 3. Save Curated Data
    try:
        df_features.to_parquet(CURATED_DATA_PATH, index=False)
        print(f"Curated dataset saved to {CURATED_DATA_PATH}. Shape: {df_features.shape}")
    except Exception as e:
        print(f"Error saving curated data: {e}")

    print("Data processing pipeline finished.")

if __name__ == "__main__":
    main()