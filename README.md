# Hybrid Adaptive Machine Learning Framework (HAMLeF)

## Overview

HAMLeF is a machine learning framework designed for financial risk prediction, with a focus on small and medium-sized enterprises (SMEs). It integrates traditional financial data with alternative sources like transaction details to deliver accurate, scalable, and interpretable predictions.

### **Key Features**
1. **Data Preprocessing**:
   - Handles missing data, scaling, and encoding.
   - Automatically removes irrelevant or unique identifier columns.
   - Saves processed data for reproducibility.

2. **Model Training**:
   - Trains a Random Forest Classifier for predicting risk categories.
   - Supports evaluation metrics like accuracy, precision, recall, and ROC-AUC.

3. **Interpretability**:
   - Uses SHAP (SHapley Additive exPlanations) to visualize feature importance and interpret model predictions.

4. **Deployment**:
   - REST API built with Flask for real-time predictions.

---

## **Project Structure**

HAMLeF/
│
├── data/
│   ├── raw/           # Contains raw data files
│   ├── processed/     # Preprocessed and cleaned data files
│
├── notebooks/
│   ├── EDA.ipynb      # Exploratory Data Analysis
│   ├── ModelTesting.ipynb  # Model experiments and testing
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── model_training.py      # Model training and evaluation
│   ├── interpretability.py    # SHAP interpretability analysis
│   ├── deployment.py          # Flask app for model deployment
│
├── models/
│   ├── model.pkl       # Trained model saved for deployment
│
├── tests/
│   ├── test_preprocessing.py  # Unit tests for preprocessing
│   ├── test_training.py       # Unit tests for training
│
├── docs/
│   ├── PRD.pdf          # Product Requirements Document
│   ├── shap_summary_plot.png  # SHAP summary plot for features
│   ├── shap_feature_importance.png  # SHAP feature importance plot
│
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-repository/hamlef.git
cd hamlef
```

2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Usage

1. Data Preprocessing

Run the preprocessing script to clean and prepare your data:

python src/data_preprocessing.py

	•	Input: Raw data file located in data/raw/.
	•	Output: Preprocessed files saved to data/processed/.

2. Model Training

Train the model using the preprocessed data:

python src/model_training.py

	•	Input: Data from data/processed/.
	•	Output:
	•	Trained model saved to models/model.pkl.
	•	Evaluation metrics printed to the console.

3. Interpretability

Generate SHAP visualizations to interpret model predictions:

python src/interpretability.py

	•	Output: SHAP summary and feature importance plots saved in docs/.

4. Deployment

Start the Flask API for real-time predictions:

python src/deployment.py

	•	Access the API at: http://127.0.0.1:5000/.

5. API Usage

Use tools like curl or Postman to interact with the API.

Example Request

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [{"feature1": 123, "feature2": 456, "feature3": "category"}]}'

Example Response

{
    "predictions": [1],
    "probabilities": [[0.3, 0.7]]
}

Evaluation Metrics

The model is evaluated on the following metrics:
	1.	Accuracy: Overall correctness of predictions.
	2.	Precision: Ratio of true positives to predicted positives.
	3.	Recall: Ratio of true positives to actual positives.
	4.	ROC-AUC: Area under the Receiver Operating Characteristic curve.

Dependencies

The project requires the following Python packages:
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib
	•	shap
	•	flask

Install these using:

pip install -r requirements.txt

Contributing

We welcome contributions! To contribute:
	1.	Fork the repository.
	2.	Create a new branch for your feature or bugfix.
	3.	Commit your changes and submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or support, please reach out to:
	•	Email: toyoski@gmail.com
	•	GitHub: matthewtech-o
