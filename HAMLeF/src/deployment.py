from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = 'models/best_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None  # Fallback in case the model is not found

# ==============================
# Home Route
# ==============================
@app.route('/')
def home():
    """Homepage with basic instructions and links."""
    return render_template('index.html')


# ==============================
# Prediction Route
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features."""
    if not model:
        return jsonify({'error': 'Model not loaded. Please check the model file.'}), 500
    
    try:
        data = request.json.get('features')
        if not data:
            return jsonify({'error': 'No input features provided.'}), 400

        # Perform prediction
        prediction = model.predict([data])
        prediction_proba = model.predict_proba([data]).tolist()
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': prediction_proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================
# Error Handling
# ==============================
@app.errorhandler(404)
def not_found_error(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Route not found.'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500


# ==============================
# About Page
# ==============================
@app.route('/about')
def about():
    """Provide information about the application."""
    return render_template('about.html')


# ==============================
# Run the Flask App
# ==============================
if __name__ == '__main__':
    app.run(debug=True)