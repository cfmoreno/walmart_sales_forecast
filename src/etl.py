import pandas as pd
import sqlalchemy
import os
# Import common utilities and paths
from utils import (
    load_csv_data, PROCESSED_PATH, DATABASE_URI, INITIAL_TABLE_NAME, DATA_PATH
)

def transform_data(train_df: pd.DataFrame, test_df: pd.DataFrame, stores_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Transform and merge data."""
    print("Transforming and merging data...")

    # Convert Date columns to datetime objects - handle potential errors
    train_df['Date'] = pd.to_datetime(train_df['Date'], errors='coerce')
    test_df['Date'] = pd.to_datetime(test_df['Date'], errors='coerce')
    features_df['Date'] = pd.to_datetime(features_df['Date'], errors='coerce')

    # Drop rows where Date could not be parsed (unlikely but safe)
    train_df.dropna(subset=['Date'], inplace=True)
    test_df.dropna(subset=['Date'], inplace=True)
    features_df.dropna(subset=['Date'], inplace=True)


    # Handle missing values in features_df (especially MarkDowns)
    # MarkDowns are typically promotions applied during specific weeks.
    # NaN often means no promotion occurred, so filling with 0 is a reasonable approach.
    # For Temperature, Fuel_Price, CPI, Unemployment, interpolation might be better
    # or carrying forward the last known value. Let's stick to ffill + fillna(0) for now.
    features_df = features_df.sort_values(by=['Store', 'Date']) # Ensure correct order for ffill
    cols_to_ffill = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in cols_to_ffill:
        if col in features_df.columns:
             features_df[col] = features_df.groupby('Store')[col].ffill()

    # After forward fill, some NaNs might remain at the start of a store's data - fill those too
    # Also fill MarkDown NaNs (assume 0 if not specified)
    markdown_cols = [f'MarkDown{i}' for i in range(1, 6)]
    cols_to_fillna_zero = cols_to_ffill + markdown_cols
    for col in cols_to_fillna_zero:
         if col in features_df.columns:
              features_df[col] = features_df[col].fillna(0)


    # Merge dataframes
    # Start by merging features with train and test
    train_merged = pd.merge(train_df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    test_merged = pd.merge(test_df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')

    # Merge with store data
    train_merged = pd.merge(train_merged, stores_df, on='Store', how='left')
    test_merged = pd.merge(test_merged, stores_df, on='Store', how='left')

    # Combine train and test for consistent processing (useful for feature engineering later)
    # Add a source column to distinguish
    train_merged['source'] = 'train'
    test_merged['source'] = 'test'
    combined_df = pd.concat([train_merged, test_merged], ignore_index=True)

    print("Transformation and merging complete.")
    return combined_df

def load_data_to_db(df: pd.DataFrame, db_uri: str, table_name: str):
    """Load data into SQLite database."""
    print(f"Loading data into {db_uri} table '{table_name}'...")
    engine = sqlalchemy.create_engine(db_uri)

    try:
        # Use 'replace' if you want to rerun the ETL and overwrite the table
        # Use 'append' if you were simulating incremental loads (more complex for this dataset initially)
        # Use 'fail' to prevent accidental overwrites
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        print("Data loading complete.")

        # Optional: Add an index for faster querying if needed later
        # with engine.connect() as conn:
        #     conn.execute(sqlalchemy.text(f'CREATE INDEX idx_{table_name}_store_date ON {table_name} ("Store", "Date");'))
        #     print(f"Index created on Store and Date for table '{table_name}'.")

    except Exception as e:
        print(f"Error loading data to database: {e}")


if __name__ == "__main__":
    # Run the ETL pipeline
    train_df, test_df, stores_df, features_df = load_csv_data(DATA_PATH)
    combined_processed_df = transform_data(train_df, test_df, stores_df, features_df)
    load_data_to_db(combined_processed_df, DATABASE_URI, INITIAL_TABLE_NAME)
    print("\nETL pipeline finished successfully!")