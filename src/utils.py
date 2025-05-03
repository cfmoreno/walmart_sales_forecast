# src/utils.py
import pandas as pd
import sqlalchemy
import os
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# No need to import actual model classes here, just related libraries

# --- Define Common Paths and Constants ---
# Update these if your folder structure changes relative to src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up two levels from src/
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'predictions')

DATABASE_NAME = 'walmart_sales.db'
DATABASE_URI = f'sqlite:///{os.path.join(PROCESSED_PATH, DATABASE_NAME)}'

# Table names within the database
INITIAL_TABLE_NAME = 'walmart_sales' # Table created by etl.py
FEATURES_TABLE_NAME = 'walmart_sales_features' # Table created by features.py

# Ensure necessary directories exist
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)


# --- Common Data Loading Functions ---
def load_csv_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract data from CSV files."""
    print("Extracting data from CSVs...")
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    stores_df = pd.read_csv(os.path.join(data_path, 'stores.csv'))
    features_df = pd.read_csv(os.path.join(data_path, 'features.csv'))
    print("Extraction complete.")
    return train_df, test_df, stores_df, features_df

def load_processed_data(db_uri: str, table_name: str) -> pd.DataFrame | None:
    """Load processed data from the database."""
    print(f"Loading data from {db_uri} table '{table_name}'...")
    engine = sqlalchemy.create_engine(db_uri)
    try:
        df = pd.read_sql_table(table_name, engine)
        # Ensure Date is datetime - handle potential errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True) # Drop rows where Date couldn't be parsed

        # Ensure Weekly_Sales is numeric and handle potential errors if the column exists
        if 'Weekly_Sales' in df.columns:
             df['Weekly_Sales'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
             # Keep rows with NaN Weekly_Sales only if source is 'test'
             # In 'train' data, NaN sales should be dropped for modeling
             initial_rows = len(df)
             df = df[~((df['source'] == 'train') & (df['Weekly_Sales'].isna()))].copy()
             if len(df) < initial_rows:
                  print(f"Dropped {initial_rows - len(df)} rows with NaN Weekly_Sales from training data.")


        print("Data loaded.")
        return df
    except sqlalchemy.exc.NoSuchTableError:
         print(f"Error: Table '{table_name}' not found in the database.")
         print("Please ensure the ETL and Feature Engineering steps have been run successfully.")
         return None
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

# --- Common Data Preparation/Feature Saving Functions ---
def prepare_data_for_modeling(df: pd.DataFrame, validation_split_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, list] | tuple[None, None, None, None, None, None]:
    """
    Prepare data for training and validation using a time-based split.

    Args:
        df: DataFrame loaded with features (must include 'source' and 'Date').
        validation_split_fraction: Fraction of dates to use for validation (from the end).

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, validation_data_full, feature_column_names)
               Returns a tuple of None if preparation fails.
    """
    print("Preparing data for modeling...")

    if df is None or df.empty:
         print("Input DataFrame is None or empty. Cannot prepare data.")
         return None, None, None, None, None, None

    if 'source' not in df.columns or 'Date' not in df.columns:
         print("Error: DataFrame must contain 'source' and 'Date' columns.")
         return None, None, None, None, None, None


    # Separate combined data back into train and test based on the 'source' column
    train_data = df[df['source'] == 'train'].copy()

    if train_data.empty:
         print("Error: No training data found (source == 'train').")
         return None, None, None, None, None, None


    # Define features (X) and target (y)
    # Drop columns that are not features or are identifiers/metadata needed later
    non_feature_cols = ['Weekly_Sales', 'source', 'Date', 'Store', 'Dept', 'IsHoliday']
    # Ensure all non_feature_cols actually exist in the DataFrame before trying to drop them
    existing_non_feature_cols = [col for col in non_feature_cols if col in train_data.columns]
    features = [col for col in train_data.columns if col not in existing_non_feature_cols]

    if not features:
         print("Error: No features identified after dropping non-feature columns.")
         return None, None, None, None, None, None

    X = train_data[features]
    y = train_data['Weekly_Sales']

    # --- Time-based Split ---
    dates = train_data['Date'].sort_values().unique()
    # Ensure there are enough dates for the split
    if len(dates) < 2:
        print("Error: Not enough unique dates in training data for time-based split.")
        return None, None, None, None, None, None

    split_index = int(len(dates) * (1 - validation_split_fraction))
    # Ensure split_index is valid
    if split_index < 1 or split_index >= len(dates):
         # Adjust split fraction if needed or take a fixed amount
         print(f"Warning: Calculated split index ({split_index}) is invalid for {len(dates)} dates.")
         if len(dates) > 10: # If enough data, just take last 10 weeks for validation
              split_date = dates[-10]
              print(f"Using last 10 weeks starting from {pd.to_datetime(split_date).strftime('%Y-%m-%d')} for validation.")
         else: # If very little data, maybe just use first/last row or skip validation
               print("Not enough data for a meaningful validation split.")
               # Return training data as validation data, or handle appropriately
               # For now, let's return None for validation if split is too problematic
               return None, None, None, None, None, None
    else:
         split_date = dates[split_index]


    X_train = X[train_data['Date'] < split_date]
    y_train = y[train_data['Date'] < split_date]
    X_val = X[train_data['Date'] >= split_date]
    y_val = y[train_data['Date'] >= split_date]

    # Keep a copy of the full validation data slice for WMAE calculation
    validation_data_full = train_data[train_data['Date'] >= split_date].copy()

    # --- Handle potential NaNs in features ---
    # Calculate mean ONLY from training data
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_val = X_val.fillna(train_means) # Use train means for validation set

    feature_column_names = features

    print("Data preparation complete.")
    return X_train, y_train, X_val, y_val, validation_data_full, feature_column_names

def prepare_test_data(df: pd.DataFrame, training_features: list) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Prepare test data for prediction, ensuring features match training data."""
    print("Preparing test data...")

    if df is None or df.empty:
         print("Input DataFrame is None or empty. Cannot prepare test data.")
         return None, None

    if 'source' not in df.columns:
         print("Error: DataFrame must contain 'source' column.")
         return None, None

    test_data = df[df['source'] == 'test'].copy()

    if test_data.empty:
         print("Error: No test data found (source == 'test').")
         return None, None


    # Select the features - MUST match the features used during training
    # Ensure all training features exist in the test data before selecting
    missing_test_cols = [col for col in training_features if col not in test_data.columns]
    if missing_test_cols:
        print(f"Error: Test data is missing features present in training data: {missing_test_cols}")
        return None, None

    X_test = test_data[training_features]

    # Handle potential NaNs in test features using mean from original training features
    # NOTE: This requires recalculating or loading training means.
    # A simple approach here is using the test data mean, BUT BE AWARE OF LEAKAGE.
    # Ideal: save train_means from prepare_data_for_modeling and load them here.
    test_means = X_test.mean() # Using test mean for simplicity HERE, NOT recommended for production
    X_test = X_test.fillna(test_means)


    # Ensure columns match exactly between training features and test features
    X_test = X_test.reindex(columns=training_features, fill_value=0)

    print("Test data preparation complete.")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {len(X_test.columns)}")
    return X_test, test_data[['Store', 'Dept', 'Date']] # Return features and identifiers

# --- Common Model Handling Functions ---
def save_model(model, model_path: str, model_filename: str):
    """Save the trained model using pickle."""
    filepath = os.path.join(model_path, model_filename)
    print(f"Saving model to {filepath}...")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model {filepath}: {e}")


def load_model(model_path: str, model_filename: str):
    """Load the trained model using pickle."""
    filepath = os.path.join(model_path, model_filename)
    print(f"Loading model from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}.")
        return None
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None

def save_features_list(features_list: list, model_path: str, filename: str):
    """Save the list of feature names used for training."""
    filepath = os.path.join(model_path, filename)
    print(f"Saving feature list to {filepath}...")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(features_list, f)
        print("Feature list saved successfully.")
    except Exception as e:
        print(f"Error saving feature list {filepath}: {e}")


def load_features_list(model_path: str, filename: str) -> list | None:
    """Load the list of feature names used for training."""
    filepath = os.path.join(model_path, filename)
    print(f"Loading feature list from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            features_list = pickle.load(f)
        print("Feature list loaded successfully.")
        return features_list
    except FileNotFoundError:
        print(f"Error: Feature list file not found at {filepath}.")
        print("Please ensure the corresponding train script was run successfully.")
        return None
    except Exception as e:
        print(f"Error loading feature list from {filepath}: {e}")
        return None

# --- Common Evaluation Function ---
def evaluate_model_performance(model, X: pd.DataFrame, y_actual: pd.Series, holiday_flags: pd.Series | None) -> dict:
    """
    Evaluate the model on provided data and return metrics.

    Args:
        model: The trained model.
        X: Feature DataFrame.
        y_actual: Actual target Series.
        holiday_flags: Series with boolean IsHoliday flags, matching index of y_actual/X (optional for WMAE).

    Returns:
        Dictionary of performance metrics. Returns empty dict if evaluation fails.
    """
    if model is None or X is None or y_actual is None or X.empty or y_actual.empty:
         print("Cannot evaluate: model or data is missing/empty.")
         return {}

    print("Evaluating model performance...")
    try:
        predictions = model.predict(X)

        # Ensure predictions are non-negative
        predictions[predictions < 0] = 0

        # Ensure actual target values are numeric
        y_actual = pd.to_numeric(y_actual, errors='coerce').fillna(0)

        # Ensure predictions and y_actual have the same length
        if len(predictions) != len(y_actual):
            print("Error: Prediction length mismatch with actual values.")
            # Attempt to align based on index if possible
            if predictions.shape == y_actual.shape: # Check shapes are compatible
                # Assume predictions are already aligned if shape matches y_actual series
                pass # Indices might still differ, but this check avoids crash
            elif len(predictions) == len(X): # If predictions match X, try aligning y_actual to X index
                 print("Attempting to align y_actual to X index.")
                 y_actual = y_actual.reindex(X.index)
                 y_actual = pd.to_numeric(y_actual, errors='coerce').fillna(0) # Re-ensure numeric after reindex
                 if len(predictions) != len(y_actual) or y_actual.isnull().any():
                      print("Alignment failed or resulted in NaNs. Metrics might be inaccurate.")
            else: # Cannot align reliably
                print("Severe length mismatch between predictions, X, and y_actual. Cannot calculate metrics.")
                return {}


        mae = mean_absolute_error(y_actual, predictions)
        mse = mean_squared_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, predictions)

        # Calculate WMAE
        wmae = np.nan # Default value if WMAE cannot be calculated
        if holiday_flags is not None and 'bool' in str(holiday_flags.dtype):
             # Ensure weights align with y_actual's index
             aligned_weights = holiday_flags.reindex(y_actual.index)
             # Drop any NaNs in weights that might appear from reindexing mismatches
             # Align y_actual, predictions, and weights by dropping NaNs across all
             temp_df = pd.DataFrame({'actual': y_actual, 'predicted': predictions, 'weights': aligned_weights})
             temp_df.dropna(inplace=True)

             if len(temp_df) > 0:
                  wmae = np.sum(temp_df['weights'] * np.abs(temp_df['actual'] - temp_df['predicted'])) / np.sum(temp_df['weights'])
             else:
                  print("Warning: No non-null data points left after aligning for WMAE calculation.")
                  wmae = np.nan # Cannot calculate WMAE

        else:
            # print("Warning: Holiday flags not provided or incorrect format for WMAE calculation.") # Too noisy
            pass # Don't print warning if holiday_flags is intentionally None


        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'WMAE': wmae
        }

        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2 Score: {r2:.2f}")
        if wmae is not np.nan:
            print(f"  WMAE: {wmae:.2f}")

        print("Evaluation complete.")
        return metrics

    except Exception as e:
         print(f"Error during model evaluation: {e}")
         return {}

# Helper Function to Make Predictions (Common to Predict Scripts)
def make_predictions(model, X_test: pd.DataFrame) -> np.ndarray:
    """Make predictions using the loaded model."""
    if model is None or X_test is None or X_test.empty:
        print("Model not loaded or test data is missing/empty. Cannot make predictions.")
        return np.array([])

    print("Making predictions...")
    try:
        predictions = model.predict(X_test)

        # Ensure predictions are non-negative
        predictions[predictions < 0] = 0

        print("Predictions made.")
        return predictions
    except Exception as e:
         print(f"Error during prediction: {e}")
         return np.array([])


# Helper Function to Save Predictions (Common to Predict Scripts)
def save_predictions(predictions: np.ndarray, identifiers_df: pd.DataFrame, predictions_path: str, filename: str):
    """Save predictions to a CSV file in submission format."""
    if len(predictions) == 0 or len(predictions) != len(identifiers_df):
        print("Error: Predictions length does not match identifiers length. Cannot save.")
        return

    filepath = os.path.join(predictions_path, filename)
    print(f"Saving predictions to {filepath}...")

    try:
        # The original competition required a specific format: Store_Dept_Date, Weekly_Sales
        # Need to reconstruct the 'Id' column
        # Ensure identifiers_df is not modified inplace if it might be used elsewhere
        submission_identifiers = identifiers_df.copy()
        submission_identifiers['Id'] = submission_identifiers['Store'].astype(str) + '_' + \
                                       submission_identifiers['Dept'].astype(str) + '_' + \
                                       submission_identifiers['Date'].dt.strftime('%Y-%m-%d')

        submission_df = pd.DataFrame({'Id': submission_identifiers['Id'], 'Weekly_Sales': predictions})

        submission_df.to_csv(filepath, index=False)
        print("Predictions saved successfully.")
    except Exception as e:
         print(f"Error saving predictions to {filepath}: {e}")

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