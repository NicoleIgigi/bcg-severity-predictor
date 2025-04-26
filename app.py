from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

# Import the prediction function from the prediction module
from prediction import predict_bcg_severity

app = Flask(__name__)

# Define the model path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_improved_bcg_severity_model.pkl')

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    patient_data = {
        'Age/Years': float(request.form.get('age', 0)),
        'Sex': request.form.get('sex', 'Unknown'),
        'Reported medication': request.form.get('medication', 'BCG Vaccine'),
        'Indication': request.form.get('indication', 'Immunization'),
        'Route of admin.': request.form.get('route', 'Unknown'),
        'Time to onset': request.form.get('time_to_onset', 'Unknown'),
        'Outcome': request.form.get('outcome', 'Unknown'),
        'Casuality Assessment conclusion': request.form.get('causality', 'Unknown'),
        '1st or 2nd Dose': request.form.get('dose_number', 'Unknown'),
        'Types of Vaccines': request.form.get('vaccine_type', 'BCG'),
        'Adverse event by location': request.form.get('event_location', 'Unknown'),
        'System Organ class affected': request.form.get('organ_system', 'Unknown'),
        'Mapped Term 1': request.form.get('mapped_term', 'Unknown'),
        'UMC report ID': int(request.form.get('report_id', 0)),
        'Country of primary source': request.form.get('country', 'Unknown'),
    }
    
    try:
        # Get prediction
        prediction = predict_bcg_severity(patient_data, MODEL_PATH)
        
        # Return result to template
        return render_template(
            'result.html',
            severity=prediction['predicted_severity'],
            confidence=f"{prediction['confidence']*100:.2f}%",
            recommendation=prediction['recommended_monitoring'],
            probabilities=prediction['probabilities']
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)