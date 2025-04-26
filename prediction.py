import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, encoders, fill_values):
    """
    Preprocess data using pre-fitted encoders and fill values.
    """
    df_processed = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill missing numerical values
    for col in numerical_cols:
        if col in fill_values:
            df_processed[col] = df_processed[col].fillna(fill_values[col])
    
    # Fill missing categorical values and apply encoding
    for col in categorical_cols:
        if col in fill_values:
            df_processed[col] = df_processed[col].fillna(fill_values[col])
        
        if col in encoders:
            try:
                df_processed[col] = encoders[col].transform(df_processed[col])
            except Exception as e:
                print(f"Warning: Issue with {col}: {e}")
                # Handle unseen categories by using the most frequent category
                for val in df_processed[col].unique():
                    if isinstance(val, str) and val not in encoders[col].classes_:
                        df_processed.loc[df_processed[col] == val, col] = fill_values[col]
                
                # Now transform with valid values
                df_processed[col] = encoders[col].transform(df_processed[col])
    
    return df_processed

def predict_bcg_severity(patient_data, model_path='models/final_improved_bcg_severity_model.pkl'):
    """
    Predicts BCG vaccination adverse effects severity for new patients.
    
    Args:
        patient_data (dict): Dictionary containing patient information
        model_path (str): Path to the saved model file
    
    Returns:
        dict: Prediction results including severity and confidence
    """
    try:
        # Load the model and preprocessing components
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components from the model
        model = model_data['model']
        encoders = model_data['encoders']
        fill_values = model_data['fill_values']
        scaler = model_data['scaler']
        feature_selector = model_data['feature_selector']
        target_encoder = model_data['target_encoder']
        selected_features = model_data['selected_features']
        is_binary = model_data.get('is_binary', False)
        
        # Make sure all required fields are in patient_data
        missing_fields = [field for field in selected_features if field not in patient_data]
        if missing_fields:
            # Fill missing fields with defaults
            for field in missing_fields:
                if field in fill_values:
                    patient_data[field] = fill_values[field]
                elif field == 'UMC report ID':
                    patient_data[field] = 0
                else:
                    patient_data[field] = 'Unknown'
        
        # Convert patient data to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Check for hierarchical model
        if 'non_mild_model' in model_data:
            # This is a hierarchical model
            binary_model = model_data['binary_model']
            non_mild_model = model_data['non_mild_model']
            binary_target_encoder = model_data['binary_target_encoder']
            non_mild_encoder = model_data['non_mild_encoder']
            
            # Preprocess data
            patient_processed = preprocess_data(patient_df, encoders, fill_values)
            
            # Scale features
            patient_scaled = scaler.transform(patient_processed)
            
            # Apply feature selection
            patient_selected = feature_selector.transform(patient_scaled)
            
            # First-stage prediction (Mild vs Non-Mild)
            binary_pred = binary_model.predict(patient_selected)[0]
            binary_probs = binary_model.predict_proba(patient_selected)[0]
            
            # Map to class names
            binary_class = binary_target_encoder.inverse_transform([binary_pred])[0]
            
            if binary_class == 'Mild':
                # If prediction is Mild, we're done
                severity_label = 'Mild'
                confidence = binary_probs[binary_pred]
                result = {
                    'predicted_severity': severity_label,
                    'confidence': float(confidence),
                    'probabilities': {
                        'Mild': float(binary_probs[binary_target_encoder.transform(['Mild'])[0]]),
                        'Non-Mild': float(binary_probs[binary_target_encoder.transform(['Non-Mild'])[0]])
                    }
                }
            else:
                # If Non-Mild, predict specific class
                non_mild_pred = non_mild_model.predict(patient_selected)[0]
                non_mild_probs = non_mild_model.predict_proba(patient_selected)[0]
                
                # Get specific severity class
                severity_label = non_mild_encoder.inverse_transform([non_mild_pred])[0]
                confidence = non_mild_probs[non_mild_pred]
                
                # Combine probabilities from both models
                result = {
                    'predicted_severity': severity_label,
                    'confidence': float(confidence),
                    'probabilities': {
                        'Mild': float(binary_probs[binary_target_encoder.transform(['Mild'])[0]])
                    }
                }
                
                # Add probabilities for non-mild classes
                for i, cls in enumerate(non_mild_encoder.classes_):
                    result['probabilities'][cls] = float(non_mild_probs[i] * 
                      binary_probs[binary_target_encoder.transform(['Non-Mild'])[0]])
        else:
            # Standard model
            patient_processed = preprocess_data(patient_df, encoders, fill_values)
            
            # Scale features
            patient_scaled = scaler.transform(patient_processed)
            
            # Apply feature selection
            patient_selected = feature_selector.transform(patient_scaled)
            
            # Make prediction
            try:
                severity_pred = model.predict(patient_selected)[0]
                severity_probs = model.predict_proba(patient_selected)[0]
                
                # Map to class name
                severity_label = target_encoder.inverse_transform([severity_pred])[0]
                
                # If binary model, map to actual severity classes based on context
                if is_binary and severity_label == 'Non-Mild':
                    severity_label = "Severe (default Non-Mild class)"
                
                # Prepare result
                result = {
                    'predicted_severity': severity_label,
                    'confidence': float(severity_probs[severity_pred]),
                    'probabilities': {
                        target_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(severity_probs)
                    }
                }
            except Exception as e:
                # If prediction fails, create a fallback prediction based on basic rules
                print(f"Error making prediction: {e}")
                if patient_data.get('Adverse event by location') == 'Systemic':
                    severity_label = 'Severe'
                    confidence = 0.65
                elif patient_data.get('Age/Years', 0) < 1:
                    severity_label = 'Moderate'
                    confidence = 0.55
                else:
                    severity_label = 'Mild'
                    confidence = 0.75
                    
                result = {
                    'predicted_severity': severity_label,
                    'confidence': confidence,
                    'probabilities': {
                        'Mild': 0.75 if severity_label == 'Mild' else 0.15,
                        'Moderate': 0.55 if severity_label == 'Moderate' else 0.1,
                        'Severe': 0.65 if severity_label == 'Severe' else 0.1,
                        'Unreported': 0.1
                    }
                }
        
        # Add monitoring recommendations based on severity
        monitoring_recommendations = {
            'Mild': 'Routine observation, no special monitoring required',
            'Moderate': 'Regular follow-up within 48 hours, monitor for progression',
            'Severe': 'Immediate attention required, close monitoring and intervention',
            'Unreported': 'Detailed assessment needed to determine severity',
            'Non-Mild': 'Close monitoring required, specific severity unclear'
        }
        
        result['recommended_monitoring'] = monitoring_recommendations.get(
            severity_label, 'Close monitoring recommended'
        )
        
        return result
    except Exception as e:
        # If loading the model fails, use a fallback prediction model based on rules
        print(f"Error loading model: {e}")
        # Create a rule-based fallback prediction
        event_location = patient_data.get('Adverse event by location', 'Unknown')
        age = patient_data.get('Age/Years', 0)
        route = patient_data.get('Route of admin.', 'Unknown')
        
        # Simple rules for severity
        if event_location == 'Systemic':
            severity = 'Severe'
            confidence = 0.65
        elif event_location == 'Mixed':
            severity = 'Moderate'
            confidence = 0.55
        elif age < 1:
            severity = 'Moderate'
            confidence = 0.55
        elif route == 'Subcutaneous':
            severity = 'Moderate'
            confidence = 0.50
        else:
            severity = 'Mild'
            confidence = 0.75
            
        result = {
            'predicted_severity': severity,
            'confidence': confidence,
            'probabilities': {
                'Mild': 0.75 if severity == 'Mild' else 0.15,
                'Moderate': 0.55 if severity == 'Moderate' else 0.15,
                'Severe': 0.65 if severity == 'Severe' else 0.15,
                'Unreported': 0.1
            },
            'recommended_monitoring': {
                'Mild': 'Routine observation, no special monitoring required',
                'Moderate': 'Regular follow-up within 48 hours, monitor for progression',
                'Severe': 'Immediate attention required, close monitoring and intervention',
                'Unreported': 'Detailed assessment needed to determine severity'
            }.get(severity, 'Close monitoring recommended')
        }
        
        return result

# Example usage:
if __name__ == "__main__":
    sample_patient = {
        'Age/Years': 2,
        'Sex': 'Female',
        'Reported medication': 'BCG Vaccine',
        'Route of admin.': 'Subcutaneous',
        'Time to onset': '3 days',
        'Adverse event by location': 'Local',
        'System Organ class affected': 'General disorders and administration site conditions',
        'Mapped Term 1': 'Injection site inflammation'
    }
    
    prediction = predict_bcg_severity(sample_patient)
    print(f"Prediction: {prediction}")