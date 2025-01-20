from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
import os

def load_data(data_path="data/processed/train_test_splits.pkl"):
    """
    Load preprocessed train and test splits saved as a joblib file.

    Parameters:
        data_path (str): Path to the joblib file containing the splits.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    return joblib.load(data_path)

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics

def save_model(model, model_path="models/best_model.pkl"):
    """Save the trained model to a file."""
    directory = os.path.dirname(model_path)
    if directory:  # Create the directory only if a directory is specified
        os.makedirs(directory, exist_ok=True)
    with open(model_path, "wb") as f:
        joblib.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_data()

    # Train the model
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC-ROC Score: {metrics['roc_auc']:.4f}")
    
    print("\nDetailed Classification Report:")
    for label, report in metrics['classification_report'].items():
        if isinstance(report, dict):  # Avoid printing metadata like 'accuracy'
            print(f"Class {label}:")
            for metric, score in report.items():
                print(f"  {metric}: {score:.4f}")

    # Save the trained model
    save_model(model,  model_path="models/best_model.pkl")