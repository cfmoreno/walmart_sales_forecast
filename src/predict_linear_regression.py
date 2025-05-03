# src/predict_linear_regression.py
import pandas as pd
import os
import pickle # Still needed for saving/loading the model object itself
import numpy as np

# Import common utilities and paths
from utils import (
    load_processed_data, load_model, load_features_list, prepare_test_data,
    make_predictions, save_predictions, DATABASE_URI, FEATURES_TABLE_NAME,
    MODEL_PATH, PREDICTIONS_PATH
)

# Model-specific filenames
MODEL_FILENAME = 'linear_regression_model.pkl'
FEATURES_FILENAME = 'linear_regression_features.pkl' # File with saved feature names
SUBMISSION_FILENAME = 'submission_lr.csv' # Unique name for predictions file

# Ensure predictions directory exists (already handled in utils, but harmless here)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)

# NOTE: prepare_test_data, make_predictions, save_predictions functions are now in utils.py

if __name__ == "__main__":
    # 1. Load feature-engineered data from the database
    feature_df = load_processed_data(DATABASE_URI, FEATURES_TABLE_NAME)

    if feature_df is not None:
        # 2. Load the list of features used during training (essential for correct columns)
        training_features = load_features_list(MODEL_PATH, FEATURES_FILENAME)

        if training_features is not None:
            # 3. Prepare the test data using the loaded feature list
            X_test, test_identifiers = prepare_test_data(feature_df, training_features)

            if X_test is not None and not X_test.empty:
                 # 4. Load the trained model
                 trained_model = load_model(MODEL_PATH, MODEL_FILENAME)

                 # 5. Make predictions on the test data if model loaded successfully
                 if trained_model is not None:
                     test_predictions = make_predictions(trained_model, X_test)

                     # 6. Save the predictions to a CSV file
                     if test_predictions is not None and len(test_predictions) > 0:
                         save_predictions(test_predictions, test_identifiers, PREDICTIONS_PATH, SUBMISSION_FILENAME)
                     else:
                         print("\nSkipping prediction saving: No predictions were generated.")
                 else:
                     print("\nSkipping prediction due to model loading failure.")
            else:
                 print("\nSkipping prediction: Test data preparation failed or resulted in empty data.")
        else:
             print("\nSkipping prediction due to failure loading feature list.")
    else:
        print("\nSkipping prediction due to data loading failure.")

    print("\nLinear Regression prediction script finished.")