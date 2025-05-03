# src/train_random_forest.py
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

# Import common utilities and paths
from utils import (
    load_processed_data, prepare_data_for_modeling, evaluate_model_performance,
    save_model, save_features_list, DATABASE_URI, FEATURES_TABLE_NAME, MODEL_PATH
)

# Model-specific filenames
MODEL_FILENAME = 'random_forest_model.pkl'
FEATURES_FILENAME = 'random_forest_features.pkl' # File to save feature names

# Ensure model directory exists (already handled in utils, but harmless here)
os.makedirs(MODEL_PATH, exist_ok=True)

def train_forecasting_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the RandomForest forecasting model."""
    print("Training the RandomForest forecasting model...")

    # Model instantiation and training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Model training complete.")
    return model

if __name__ == "__main__":
    # 1. Load feature-engineered data from the database
    feature_df = load_processed_data(DATABASE_URI, FEATURES_TABLE_NAME)

    if feature_df is not None:
        # 2. Prepare data (split train/val, get features list, handle NaNs)
        # validation_split_fraction=0.2 means use the last 20% of dates for validation
        X_train, y_train, X_val, y_val, validation_data_full, feature_column_names = prepare_data_for_modeling(feature_df, validation_split_fraction=0.2)

        # Check if data preparation was successful
        if X_train is not None and not X_train.empty:
             # 3. Train the model using the training data
             trained_model = train_forecasting_model(X_train, y_train)

             # 4. Evaluate the model on validation data
             print("\nEvaluating on Validation Set:")
             # Pass the 'IsHoliday' flags from the validation data for WMAE calculation
             validation_holiday_flags = validation_data_full['IsHoliday'] if validation_data_full is not None else None
             metrics = evaluate_model_performance(trained_model, X_val, y_val, validation_holiday_flags)

             # 5. Save the trained model
             save_model(trained_model, MODEL_PATH, MODEL_FILENAME)

             # 6. Save the list of feature names used for training
             save_features_list(feature_column_names, MODEL_PATH, FEATURES_FILENAME)

             print("\nRandom Forest model training script finished successfully!")
        else:
             print("\nSkipping training: Data preparation failed or resulted in empty training data.")
    else:
        print("\nSkipping training due to data loading failure.")