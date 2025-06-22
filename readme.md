# Walmart Sales Forecast
![Screenshot 2025-06-08 164908](https://github.com/user-attachments/assets/41e9fcb6-5daa-4232-9331-4ed26efbd3df)

## 📊 Data Source

The dataset used for this project is the [Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data) dataset available on Kaggle. It provides historical weekly sales data for 45 stores across various departments, along with store details and external economic factors.

## 📈 Methodology & Pipeline

The project follows a structured, sequential pipeline:

1.  **Extraction & Initial Loading (`src/etl.py`):**
    *   Raw CSV files (`train.csv`, `test.csv`, `stores.csv`, `features.csv`) are read into pandas DataFrames.
    *   Initial cleaning (date format conversion) and merging of the different data sources are performed.
    *   Missing values in external features (especially MarkDowns) are handled using forward fill and imputation (filling NaNs with 0).
    *   A `source` column is added to distinguish between original train and test data points.
    *   The combined, cleaned dataset is loaded into a SQLite database (`data/processed/walmart_sales.db`) as a table (`walmart_sales`), establishing a structured data layer.

2.  **Feature Engineering (`src/features.py`):**
    *   Data is loaded from the `walmart_sales` table in the DB.
    *   Time-series specific features are created programmatically: Year, Month, Week, Day, DayOfWeek, WeekOfYear.
    *   Lagged features (previous weeks' values) for external factors are generated per store.
    *   Rolling window statistics (mean, standard deviation) for relevant features are calculated per store.
    *   Categorical features (`Store Type`) are one-hot encoded.
    *   Remaining NaNs introduced by feature creation (lags/rolling windows at the start of series) are imputed (using ffill/bfill within groups and filling initial NaNs with 0).
    *   The resulting DataFrame with all features is loaded into a new table (`walmart_sales_features`) in the same SQLite database.

3.  **Data Analysis (`src/analyze.py`):**
    *   Data is loaded from the `walmart_sales` table (the initial cleaned data).
    *   Various analyses are conducted using matplotlib and seaborn to gain insights:
        *   Overall sales trends over time with holiday highlighting.
        *   Distribution of sales during holiday vs. non-holiday weeks (using box plots).
        *   Analysis of specific holiday impact.
        *   Investigation of negative sales records: frequency, location (Store/Dept), potential correlation with events.
        *   Comparison of sales performance across different store types and sizes.
        *   Identification and visualization of sales trends for top and bottom performing stores.
        *   Analysis of overall department performance.
        *   Correlation analysis between external features (Temperature, Fuel Price, CPI, Unemployment, Markdowns) and total sales.

4.  **Model Training (`src/train_*.py`):**
    *   Separate scripts are used for each model: `train_random_forest.py`, `train_linear_regression.py`, `train_xgboost.py`.
    *   Each script loads data from the `walmart_sales_features` table.
    *   Data is split into training and validation sets using a **time-based split** (last 20% of dates for validation) to simulate real-world forecasting.
    *   The specified model is trained on the training data.
    *   The trained model is evaluated on the validation set using MAE, MSE, RMSE, R2, and the competition's WMAE metric.
    *   The trained model object is saved to disk (`models/`) using `pickle`.
    *   Crucially, the list of feature names used during training is also saved (`models/*_features.pkl`) to ensure prediction uses the exact same feature set.

5.  **Model Comparison (`src/compare_models.py`):**
    *   Data is loaded from the `walmart_sales_features` table.
    *   The same time-based validation split logic from the training scripts is replicated to isolate the validation data.
    *   Each trained model (`random_forest_model.pkl`, `linear_regression_model.pkl`, `xgboost_model.pkl`) is loaded.
    *   The validation features are prepared for each model by selecting the exact columns they were trained on (using the saved feature lists).
    *   Predictions are made by each model on the validation data.
    *   Performance metrics (MAE, RMSE, R2, WMAE) are calculated for each model on the validation set.
    *   Results are printed in a comparison table, sorted by WMAE, to identify the best performing model.

6.  **Prediction (`src/predict_*.py`):**
    *   Separate scripts (`predict_random_forest.py`, `predict_linear_regression.py`, `predict_xgboost.py`) load the respective trained model and its feature list.
    *   The original test data rows are filtered from the feature-engineered DataFrame.
    *   Test features are prepared, ensuring the exact same columns as used in training are selected and in the correct order.
    *   Predictions are made on the prepared test data.
    *   Predictions are saved into unique CSV files (`predictions/submission_rf.csv`, etc.) in the format required for the original Kaggle competition (Id, Weekly_Sales).

## 🔍 Key Analyses and Insights

Based on the analysis (`notebooks/first_exploration.py`), key insights observed include:

*   **Pronounced Holiday Peaks:** Sales exhibit significant spikes during specific holiday weeks (Thanksgiving, Christmas), confirming the importance of the `IsHoliday` flag and surrounding features.
*   **Store Type Performance:** Store Type A generally shows higher average sales and different sales patterns compared to Types B and C, often correlating with larger store sizes.
*   **Negative Sales Investigation:** Analysis of negative sales records reveals they occur across various stores and departments, potentially indicating concentrated return activity or data anomalies around certain periods.
*   **Departmental Variation:** Sales performance varies significantly by department, with a distinct set of top-performing departments contributing a large portion of overall sales.
*   **External Factor Influence:** Visualizations and correlations suggest potential relationships between external factors like Temperature, CPI, and Fuel Price with overall sales trends, though these relationships are not always straightforward and can vary by store.

These insights provide valuable context for understanding the sales data and inform the feature engineering process for forecasting.

## 🏆 Model Comparison Results

After training and evaluating Random Forest, Linear Regression, and XGBoost models on the validation set, the performance metrics based on WMAE (Weighted Mean Absolute Error - lower is better) were compared.

--- Model Comparison Results (Validation Set) ---
                        MAE      RMSE      WMAE    R2
Random Forest      12984.07  21822.43  13707.73  0.01
XGBoost            13484.77  21355.47  13736.77  0.05
Linear Regression  14222.59  21227.94  15555.27  0.07

Best performing model based on WMAE: Random Forest (WMAE: 13707.73)

indicating it is the most accurate predictor of weekly sales, particularly sensitive to capturing holiday sales fluctuations due to the weighting. This aligns with expectations, as tree-based models like Random Forest and XGBoost are typically better at capturing non-linear relationships and interactions in the features compared to linear models.

## ⚙️ How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cfmoreno/walmart_sales_forecast.git
    cd walmart_sales_forecast # Navigate to the project root directory
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On macOS/Linux
    source .venv/bin/activate
    # On Windows (Command Prompt)
    .venv\Scripts\activate.bat
    # On Windows (PowerShell - might need execution policy fix)
    # .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the data:** Download the CSV files (`train.csv`, `test.csv`, `stores.csv`, `features.csv`) from the [Kaggle dataset page](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data) and place them in the `data/raw/` directory within your project.
5.  **Run the ETL pipeline:** Processes raw data and loads it into the SQLite DB.
    ```bash
    python src/etl.py
    ```
6.  **Run the Feature Engineering script:** Adds features and loads them to the DB.
    ```bash
    python src/features.py
    ```
7.  **Run the Analysis script (Optional but recommended):** Generates EDA plots and prints key insights.
    ```bash
    python src/analyze.py
    ```
8.  **Train the Forecasting Models:** Trains all three models and saves them along with their feature lists.
    ```bash
    python src/train_random_forest.py
    python src/train_linear_regression.py
    python src/train_xgboost.py
    ```
9.  **Compare the Models:** Evaluates models on the validation set and prints performance comparisons.
    ```bash
    python src/compare_models.py
    ```
10. **Generate Predictions:** Uses the trained models to predict sales on the original test data and saves submission files.
    ```bash
    python src/predict_random_forest.py # Generates predictions/submission_rf.csv
    python src/predict_linear_regression.py # Generates predictions/submission_lr.csv
    python src/predict_xgboost.py # Generates predictions/submission_xgb.csv
    ```

## ⏭️ Future Enhancements

*   Implement more advanced time-series feature engineering (e.g., cyclical features for week/month, interaction terms, external API data for future features like weather forecasts).
*   Perform hyperparameter tuning for the models (especially Random Forest and XGBoost) using techniques like Grid Search or Random Search with time-series cross-validation.
*   Explore alternative time series specific models (e.g., Prophet, ARIMA variants, deep learning models like LSTMs).
*   Add more sophisticated data quality checks within the ETL pipeline and log potential issues.
*   Containerize the application using Docker for easier deployment and reproducibility.
*   Implement a simple API (e.g., using Flask or FastAPI) to serve predictions from the best-performing model.
*   Explore integrating a workflow orchestration tool like Apache Airflow or Prefect to manage the pipeline runs.
*   Utilize a more robust database (like PostgreSQL) or cloud data warehouse (like Snowflake or BigQuery) for larger scale data.

## ✉️ Contact

*   **Name:** Carlos Moreno
*   **LinkedIn:** [https://www.linkedin.com/in/cfmoreno/]
*   **Email:** [cfmorenod@gmail.com]

---

Feel free to connect with me to discuss this project or opportunities in Data Engineering, Data Science, or Data Analysis!
