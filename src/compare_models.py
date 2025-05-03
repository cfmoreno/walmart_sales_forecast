# src/compare_models.py
import pandas as pd
import os
import pickle
import numpy as np

# Import common utilities and paths
from utils import (
    load_processed_data, load_model, load_features_list, evaluate_model_performance,
    DATABASE_URI, FEATURES_TABLE_NAME, MODEL_PATH
)

# Define models and their corresponding filenames (Keep this list local as it defines the comparison)
models_to_compare = {
    'Random Forest': {
        'model_file': 'random_forest_model.pkl',
        'features_file': 'random_forest_features.pkl' # Match filenames used in train scripts
    },
    'Linear Regression': {
        'model_file': 'linear_regression_model.pkl',
        'features_file': 'linear_regression_features.pkl'
    },
    'XGBoost': {
        'model_file': 'xgboost_model.pkl',
        'features_file': 'xgboost_features.pkl'
    }
}

# Define the validation split fraction (MUST match what was used in prepare_data_for_modeling)
VALIDATION_SPLIT_FRACTION = 0.2


def get_validation_data_split(df: pd.DataFrame, validation_split_fraction: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | tuple[None, None, None]:
    """
    Replicate the validation data split from the training scripts.

    Args:
        df: DataFrame loaded with features (must include 'source' and 'Date').
        validation_split_fraction: Fraction of dates used for validation (from the end).

    Returns:
        Tuple: (X_val_full_features, y_val, validation_data_full)
               Returns a tuple of None if splitting fails.
    """
    print(f"Replicating validation data split ({validation_split_fraction*100}% from the end)...")

    if df is None or df.empty:
         print("Input DataFrame is None or empty. Cannot split data.")
         return None, None, None

    if 'source' not in df.columns or 'Date' not in df.columns:
         print("Error: DataFrame must contain 'source' and 'Date' columns.")
         return None, None, None

    train_data = df[df['source'] == 'train'].copy()

    if train_data.empty:
         print("Error: No training data found (source == 'train').")
         return None, None, None

    dates = train_data['Date'].sort_values().unique()
    if len(dates) < 2:
        print("Error: Not enough unique dates in training data for time-based split.")
        return None, None, None

    split_index = int(len(dates) * (1 - validation_split_fraction))
    if split_index < 1 or split_index >= len(dates):
         print(f"Warning: Calculated split index ({split_index}) is invalid for {len(dates)} dates.")
         if len(dates) > 10:
              split_date = dates[-10]
              print(f"Using last 10 weeks starting from {pd.to_datetime(split_date).strftime('%Y-%m-%d')} for validation.")
         else:
              print("Not enough data for a meaningful validation split.")
              return None, None, None
    else:
         split_date = dates[split_index]

    # Select the validation slice of the original training data
    validation_data_full = train_data[train_data['Date'] >= split_date].copy()

    if validation_data_full.empty:
         print(f"Error: Validation data slice is empty starting from {pd.to_datetime(split_date).strftime('%Y-%m-%d')}.")
         return None, None, None

    # Separate features (including all potential features created in features.py) and target
    # We need all features here so we can subset them based on the saved feature lists later
    non_feature_cols_potential = ['Weekly_Sales', 'source'] # Only exclude these for this initial split
    X_val_full_features = validation_data_full.drop(columns=[col for col in non_feature_cols_potential if col in validation_data_full.columns])
    y_val = validation_data_full['Weekly_Sales'] # Actual target values


    print(f"Validation set size: {len(validation_data_full)}")
    print("Validation data prepared.")
    return X_val_full_features, y_val, validation_data_full # Return features, target, and full data

def prepare_features_for_validation_per_model(validation_data_full_features: pd.DataFrame, training_features: list) -> pd.DataFrame | None:
     """
     Select and preprocess features for the validation set to match a specific model's training features.
     This replicates the feature selection and imputation steps from prepare_data_for_modeling
     specifically for the validation subset and a given model's feature list.
     """
     print("Preparing validation features for specific model...")

     if validation_data_full_features is None or validation_data_full_features.empty:
          print("Input validation features DataFrame is None or empty.")
          return None

     # Select only the features used by this specific model during training
     # Ensure all required training features exist in the validation data features
     missing_val_cols = [col for col in training_features if col not in validation_data_full_features.columns]
     if missing_val_cols:
         print(f"Error: Validation data is missing features present in training features list: {missing_val_cols}")
         return None

     X_val = validation_data_full_features[training_features].copy()

     # Handle potential NaNs in validation features
     # NOTE: For comparison, using the validation mean is acceptable.
     # For actual prediction, use the training mean saved during training.
     val_means = X_val.mean()
     X_val = X_val.fillna(val_means)


     # Ensure columns match exactly
     X_val = X_val.reindex(columns=training_features, fill_value=0)

     print("Validation features prepared.")
     return X_val


if __name__ == "__main__":
    print("Starting model comparison...")

    # 1. Load the full feature-engineered dataset
    feature_df = load_processed_data(DATABASE_URI, FEATURES_TABLE_NAME)

    if feature_df is None:
        print("Failed to load feature data. Cannot compare models.")
        exit() # Exit if data loading fails

    # 2. Replicate the validation data split (get the full validation data slice and target)
    X_val_full_features, y_val, validation_data_full = get_validation_data_split(feature_df, VALIDATION_SPLIT_FRACTION)

    if X_val_full_features is None or y_val is None or validation_data_full is None:
         print("Failed to prepare validation data split. Cannot compare models.")
         exit()

    comparison_results = {}

    # 3. Loop through each model, load it, make predictions, and evaluate
    for model_name, info in models_to_compare.items():
        print(f"\n--- Comparing {model_name} ---")

        # Load the list of features used by this model during training
        training_features = load_features_list(MODEL_PATH, info['features_file'])

        if training_features is None:
            print(f"Skipping {model_name}: Could not load features list.")
            continue # Skip to the next model

        # Prepare the validation features specifically for this model's feature set
        X_val_prepared = prepare_features_for_validation_per_model(X_val_full_features, training_features)

        if X_val_prepared is None:
             print(f"Skipping {model_name}: Failed to prepare validation features.")
             continue

        # Load the trained model
        model = load_model(MODEL_PATH, info['model_file'])

        if model is None:
            print(f"Skipping {model_name}: Could not load model.")
            continue # Skip to the next model

        # Evaluate the loaded model on the prepared validation data
        # Pass the 'IsHoliday' column from the full validation data for WMAE calculation
        validation_holiday_flags = validation_data_full['IsHoliday']
        metrics = evaluate_model_performance(model, X_val_prepared, y_val, validation_holiday_flags)

        comparison_results[model_name] = metrics # Store metrics


    # 4. Display Comparison Results
    print("\n--- Model Comparison Results (Validation Set) ---")
    if comparison_results:
        # Convert results to a pandas DataFrame for nice printing
        results_df = pd.DataFrame(comparison_results).T # Transpose to have models as rows
        # Ensure WMAE is treated as a number for sorting, handling potential NaN
        results_df['WMAE_sort'] = results_df['WMAE'].replace({np.nan: np.inf})
        # Sort by WMAE (lower is better)
        results_df = results_df.sort_values(by='WMAE_sort')
        results_df = results_df.drop(columns='WMAE_sort') # Drop the sorting column

        print(results_df[['MAE', 'RMSE', 'WMAE', 'R2']].round(2)) # Print key metrics, rounded

        # Highlight the best performing model based on WMAE
        if 'WMAE' in results_df.columns and not results_df['WMAE'].isnull().all():
            best_model_name = results_df['WMAE'].idxmin()
            best_wmae = results_df['WMAE'].min()
            print(f"\nBest performing model based on WMAE: {best_model_name} (WMAE: {best_wmae:.2f})")
        else:
             print("\nCould not determine best model based on WMAE (WMAE results missing or all NaN).")

    else:
        print("No models were successfully compared.")

    print("\nModel comparison script finished.")