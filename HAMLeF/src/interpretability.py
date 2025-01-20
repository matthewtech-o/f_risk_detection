import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_data(data_path="data/processed/train_test_splits.pkl"):
    """
    Load preprocessed test data from a joblib file.
    Parameters:
        data_path (str): Path to the joblib file containing train-test splits.
    Returns:
        pd.DataFrame: Test data (X_test).
    """
    _, X_test, _, _ = joblib.load(data_path)
    return X_test

def load_model(model_path="models/best_model.pkl"):
    """
    Load the trained model from a file.
    Parameters:
        model_path (str): Path to the saved model file.
    Returns:
        sklearn model: The trained model.
    """
    return joblib.load(model_path)

def explain_model(model, X_data, subsample_size=10000):
    """
    Generate SHAP values using a subsample of the data for performance.
    Parameters:
        model (sklearn model): Trained model.
        X_data (pd.DataFrame): Input features for SHAP.
        subsample_size (int): Number of rows to sample for SHAP computation.
    Returns:
        shap_values: SHAP values for the subsample.
        explainer: SHAP explainer object.
        X_sample: Subsampled data for which SHAP values were computed.
    """
    # Subsample the data
    X_sample = X_data.sample(n=subsample_size, random_state=42)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sample)
    return shap_values, explainer, X_sample

def plot_global_importance(shap_values, X_sample, output_path="docs/shap_global_importance.png"):
    """
    Generate and save SHAP global importance plot.
    Parameters:
        shap_values: SHAP values computed for the data.
        X_sample (pd.DataFrame): Subsampled data used for SHAP.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    # Adjust for multiclass or binary classification
    if isinstance(shap_values, list):  # Multiclass model
        shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
    else:  # Binary classification
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Global feature importance plot saved to {output_path}")

def plot_local_explanation(explainer, shap_values, X_sample, index=0, output_path="docs/shap_local_explanation.png"):
    """
    Generate and save SHAP local explanation plot for a single prediction.
    Parameters:
        explainer: SHAP explainer object.
        shap_values: SHAP values computed for the data.
        X_sample (pd.DataFrame): Subsampled data used for SHAP.
        index (int): Index of the data point to explain.
        output_path (str): Path to save the plot.
    """
    base_value = explainer.expected_value
    instance_shap_values = shap_values[index] if not isinstance(shap_values, list) else shap_values[1][index]

    # Multi-class or binary check
    if isinstance(base_value, list):  # Multi-class model
        base_value = base_value[1]  # Use class 1 for explanation
        instance_shap_values = shap_values[1][index]

    shap.force_plot(
        base_value,
        instance_shap_values,
        X_sample.iloc[index],
        matplotlib=True
    )
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Local explanation plot for index {index} saved to {output_path}")

if __name__ == "__main__":
    # Load data and model
    X_test = load_data()
    model = load_model()

    # Explain model using a subsample of 10,000 rows
    print("Generating SHAP values. This may take a while for large datasets...")
    shap_values, explainer, X_sample = explain_model(model, X_test)

    # Plot global importance
    plot_global_importance(shap_values, X_sample)

    # Optionally, plot a local explanation for a specific instance
    # plot_local_explanation(explainer, shap_values, X_sample, index=0)