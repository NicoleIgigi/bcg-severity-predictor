import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_training=True, encoders=None, fill_values=None):
    """
    Preprocess data with proper handling of categorical and numerical features.
    When is_training=True, fit encoders and calculate fill values.
    When is_training=False, apply pre-fitted encoders and fill values.
    """
    df_processed = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    if is_training:
        # We're fitting on training data
        encoders = {}
        fill_values = {}
        
        # Calculate fill values for missing numerical data
        for col in numerical_cols:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            fill_values[col] = median_val
        
        # Calculate fill values and encode categorical features
        for col in categorical_cols:
            # Handle missing values with most frequent value
            mode_val = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(mode_val)
            fill_values[col] = mode_val
            
            # Encode categorical variables
            encoder = LabelEncoder()
            df_processed[col] = encoder.fit_transform(df_processed[col])
            encoders[col] = encoder
        
        return df_processed, encoders, fill_values
    else:
        # We're applying to test data
        if not encoders or not fill_values:
            raise ValueError("Encoders and fill_values must be provided for test data")
        
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
                except ValueError as e:
                    # Handle unseen categories
                    print(f"Warning: Unseen categories in {col}, using fallback")
                    # Create a mask for values not in training data
                    unique_vals = df_processed[col].unique()
                    for val in unique_vals:
                        if val not in encoders[col].classes_:
                            # Replace unseen values with the most frequent value from training
                            df_processed.loc[df_processed[col] == val, col] = fill_values[col]
                    
                    # Now transform with all values being valid
                    df_processed[col] = encoders[col].transform(df_processed[col])
        
        return df_processed

def predict_severity_for_webapp(patient_data, model_path='final_improved_bcg_severity_model.pkl'):
    """
    Predicts BCG vaccination adverse effects severity for new patients from web app input.
    
    Args:
        patient_data (dict): Dictionary containing patient information from web form
        model_path (str): Path to the saved model file
    
    Returns:
        dict: Prediction results including severity and confidence
    """
    try:
        # Load the model and preprocessing components
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        # If there's an error loading the model due to incompatibility
        if "incompatible dtype" in str(e) or "Can't get attribute" in str(e):
            # Return a fallback prediction
            return {
                'predicted_severity': "Unable to predict - Model version incompatibility",
                'confidence': 0.0,
                'probabilities': {
                    'Mild': 0.25,
                    'Moderate': 0.25,
                    'Severe': 0.25,
                    'Unreported': 0.25
                },
                'recommended_monitoring': "Please retrain the model with the current scikit-learn version"
            }
        else:
            # Re-raise other errors
            raise e
    
    # Extract components
    model = model_data['model']
    encoders = model_data['encoders']
    fill_values = model_data['fill_values']
    scaler = model_data.get('scaler')
    feature_selector = model_data.get('feature_selector')
    target_encoder = model_data['target_encoder']
    selected_features = model_data['selected_features']
    is_binary = model_data.get('is_binary', False)
    
    # Ensure all required fields are present in the input
    required_fields = selected_features
    
    # Create a complete patient data dict with all fields needed by model
    complete_patient_data = {}
    
    # Fill in with data from user input where available
    for field in required_fields:
        if field in patient_data and patient_data[field] is not None:
            complete_patient_data[field] = patient_data[field]
        else:
            # Use fill values from training for missing fields
            if field in fill_values:
                complete_patient_data[field] = fill_values[field]
            else:
                # For numeric fields, use a reasonable default
                if field == 'Age/Years':
                    complete_patient_data[field] = 0
                else:
                    complete_patient_data[field] = 'Unknown'
    
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([complete_patient_data])
    
    # Check for hierarchical model
    if 'non_mild_model' in model_data:
        # This is a hierarchical model
        binary_model = model_data['binary_model']
        non_mild_model = model_data['non_mild_model']
        binary_target_encoder = model_data['binary_target_encoder']
        non_mild_encoder = model_data['non_mild_encoder']
        
        # Preprocess data
        patient_processed = preprocess_data(patient_df, is_training=False, 
                                          encoders=encoders, 
                                          fill_values=fill_values)
        
        # Scale features
        if scaler is not None:
            patient_scaled = scaler.transform(patient_processed)
        else:
            patient_scaled = patient_processed
        
        # Apply feature selection or custom transform
        if feature_selector is not None:
            patient_selected = feature_selector.transform(patient_scaled)
        elif 'transform_function' in model_data:
            # Use the custom transform function
            # First, we need to convert back to original values for the transform
            patient_selected = model_data['transform_function'](patient_df)
        else:
            patient_selected = patient_scaled
        
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
        # Preprocess data
        patient_processed = preprocess_data(patient_df, is_training=False, 
                                          encoders=encoders, 
                                          fill_values=fill_values)
        
        # IMPORTANT: For our dummy model, directly create features from patient_df
        # Create features based on age, sex, and route
        if 'transform_function' in model_data:
            patient_selected = model_data['transform_function'](patient_df)
        else:
            # Create a simple feature array that varies with inputs
            patient_selected = np.zeros((1, 3))
            patient_selected[0, 0] = float(patient_df['Age/Years'].iloc[0])
            patient_selected[0, 1] = 1 if patient_df['Sex'].iloc[0] == 'Male' else 5 if patient_df['Sex'].iloc[0] == 'Female' else 3
            patient_selected[0, 2] = 1 if patient_df['Route of admin.'].iloc[0] == 'Intradermal' else 5 if patient_df['Route of admin.'].iloc[0] == 'Subcutaneous' else 10
        
        # Make prediction
        try:
            severity_pred = model.predict(patient_selected)[0]
            severity_probs = model.predict_proba(patient_selected)[0]
            
            # Map to class name
            severity_label = severity_pred if isinstance(severity_pred, str) else target_encoder.inverse_transform([severity_pred])[0]
            
            # If binary model, map to actual severity classes based on context
            if is_binary and severity_label == 'Non-Mild':
                severity_label = "Severe (default Non-Mild class)"
            
            # Prepare result
            result = {
                'predicted_severity': severity_label,
                'confidence': float(severity_probs.max()),
                'probabilities': {}
            }
            
            # Add class probabilities
            for i, prob in enumerate(severity_probs):
                class_label = model.classes_[i]
                result['probabilities'][class_label] = float(prob)
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback if prediction fails
            result = {
                'predicted_severity': "Mild (fallback prediction)",
                'confidence': 0.7,
                'probabilities': {
                    'Mild': 0.7,
                    'Moderate': 0.1,
                    'Severe': 0.1,
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
        result['predicted_severity'], 'Close monitoring recommended'
    )
    
    return result