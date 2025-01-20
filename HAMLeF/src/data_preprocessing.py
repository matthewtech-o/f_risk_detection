import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(file_path, target_col='PAST_DUE_DAYS', test_size=0.2, random_state=42):
    """
    Preprocesses and engineers features for the dataset.

    Parameters:
        file_path (str): Path to the input CSV file.
        target_col (str): Target column for prediction.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed and split datasets.
    """
    data = pd.read_csv(file_path)

    # Convert date columns
    date_columns = ['ACCT_OPN_DATE', 'RELATIONSHIP_STARTDATE', 'DATE_OF_INCORPORATION']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    current_date = pd.to_datetime('today')
    if 'ACCT_OPN_DATE' in data.columns:
        data['LOAN_AGE'] = (current_date - data['ACCT_OPN_DATE']).dt.days
    if 'RELATIONSHIP_STARTDATE' in data.columns:
        data['RELATIONSHIP_AGE'] = (current_date - data['RELATIONSHIP_STARTDATE']).dt.days
    data.drop(columns=date_columns, inplace=True, errors='ignore')

    # Drop identifier columns
    identifier_columns = ['CORP_KEY', 'FORACID', 'SOL_ID']
    data.drop(columns=identifier_columns, inplace=True, errors='ignore')

    # Fill missing values
    transaction_columns = [col for col in data.columns if 'TRAN_AMT' in col or 'TRAN_CNT' in col]
    data[transaction_columns] = data[transaction_columns].fillna(0)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Encode categorical columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Feature engineering
    data['TOTAL_TRAN_AMT'] = data[transaction_columns].sum(axis=1)
    data['AVG_TRAN_AMT'] = data[transaction_columns].mean(axis=1)
    data['DEBIT_CREDIT_RATIO'] = (data['TOTAL_TRAN_AMT'] / (data['TOTAL_TRAN_AMT'] + 1e-6))

    # Cap outliers
    def cap_outliers(series, lower_percentile=0.01, upper_percentile=0.99):
        lower = series.quantile(lower_percentile)
        upper = series.quantile(upper_percentile)
        return np.clip(series, lower, upper)
    for col in numeric_columns:
        data[col] = cap_outliers(data[col])

    # Scale numeric features
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Handle class imbalance
    if target_col in data.columns:
        X = data.drop(columns=[target_col])
        y = data[target_col].apply(lambda x: 1 if x > 0 else 0)
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save the splits
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, "train_test_splits.pkl"))
    print(f"Preprocessing complete. Data saved as 'data/processed/train_test_splits.pkl'")
    return X_train, X_test, y_train, y_test

# Run the preprocessing script
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("data/raw/hamlef_data.csv")