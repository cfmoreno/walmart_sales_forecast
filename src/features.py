# src/features.py
import pandas as pd
import sqlalchemy
import os
import numpy as np
# Import common utilities and paths
from utils import (
    load_processed_data, load_data_to_db, DATABASE_URI, INITIAL_TABLE_NAME, FEATURES_TABLE_NAME
)


def create_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Create time series and other features."""
    print("Creating features...")

    if df is None or df.empty:
         print("Input DataFrame is None or empty. Cannot create features.")
         return None

    # Sort data - crucial for lag and rolling features
    df = df.sort_values(by=['Store', 'Dept', 'Date']).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    # Use dt.isocalendar().week and handle potential edge cases (week 53) and type
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    # Handle week 53 issue if present in the data and needed for cyclic features etc.
    # Simple approach: cap at 52 or use a cyclical feature encoding if model supports
    # For now, let's just use the int week number directly.
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['WeekOfYear'] = df['Week'] # Alias for clarity


    # Lag features for external data (Temperature, Fuel_Price etc.) - for combined data
    # Ensure these columns exist
    external_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    existing_external_cols = [col for col in external_cols if col in df.columns]

    for col in existing_external_cols:
         for lag in [1, 2, 3, 4]: # Example: lags of 1, 2, 3, 4 weeks
             df[f'{col}_Lag_{lag}'] = df.groupby('Store')[col].shift(lag)

    # Rolling features (e.g., rolling mean of temperature)
    rolling_cols = ['Temperature', 'Fuel_Price']
    existing_rolling_cols = [col for col in rolling_cols if col in df.columns]

    for col in existing_rolling_cols:
         for window in [4, 12]: # Example: 4-week and 12-week rolling mean/std
             # Use .copy() to avoid SettingWithCopyWarning with rolling operations
             df[f'{col}_RollingMean_{window}'] = df.groupby('Store')[col].rolling(window=window).mean().reset_index(level=0, drop=True).copy()
             df[f'{col}_RollingStd_{window}'] = df.groupby('Store')[col].rolling(window=window).std().reset_index(level=0, drop=True).fillna(0).copy() # Fill std=0 for window < 2


    # Fill NaNs created by lags/rolling windows (e.g., first few rows for each store)
    # ffill within groups is often better for time series, followed by filling remaining (start) NaNs.
    # Apply imputation only to the potentially new columns (lag/rolling) or check all feature columns.
    # Let's re-apply ffill and fillna(0) to relevant columns after creating new ones.
    cols_to_impute = [c for c in df.columns if c not in ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday', 'Type', 'Size', 'source']]
    for col in cols_to_impute:
         if col in df.columns:
              # Ensure the column is numeric before imputation
              df[col] = pd.to_numeric(df[col], errors='coerce')
              # Impute NaNs created by shifting/rolling
              df[col] = df.groupby('Store')[col].ffill() # Forward fill within store groups
              df[col] = df.groupby('Store')[col].bfill() # Backward fill for initial NaNs in groups
              df[col] = df[col].fillna(0) # Fill any remaining NaNs (e.g., if an entire store/col is NaN)


    # One-hot encode categorical features that will be used in the model
    # 'Type' is the main one we'll encode
    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'], prefix='Store_Type', drop_first=True) # drop_first avoids multicollinearity


    print("Feature creation complete.")
    return df

if __name__ == "__main__":
    # Run the feature engineering pipeline
    # Load the data from the initial ETL table
    processed_df = load_processed_data(DATABASE_URI, INITIAL_TABLE_NAME)

    # Create features
    features_df = create_features(processed_df)

    # Load data with features into a new table in the database
    if features_df is not None:
        load_data_to_db(features_df, DATABASE_URI, FEATURES_TABLE_NAME)
        print("\nFeature engineering pipeline finished successfully!")

        # Display info about the new features dataframe by loading it back
        print("\nFeatures DataFrame Info (Loaded from DB):")
        loaded_features_df = load_processed_data(DATABASE_URI, FEATURES_TABLE_NAME)
        if loaded_features_df is not None:
             print(loaded_features_df.info())
             print("\nSample rows with new features:")
             print(loaded_features_df.head())

             # Verify some features were created and imputed
             sample_cols = ['Date', 'Store', 'Weekly_Sales', 'Temperature', 'Temperature_Lag_1', 'Temperature_RollingMean_4']
             # Add encoded type columns if they exist
             if 'Store_Type_B' in loaded_features_df.columns: sample_cols.append('Store_Type_B')
             if 'Store_Type_C' in loaded_features_df.columns: sample_cols.append('Store_Type_C')

             existing_sample_cols = [col for col in sample_cols if col in loaded_features_df.columns]
             if existing_sample_cols:
                 print("\nChecking sample columns with created features and imputation:")
                 print(loaded_features_df[existing_sample_cols].head(10)) # Show a few rows to see lags
             else:
                 print("\nSample check columns not found.")

    else:
        print("\nSkipping feature engineering due to data loading failure.")