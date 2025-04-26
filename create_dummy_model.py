import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Create a dummy RandomForest model that can be properly serialized
dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)

# Create some dummy data that will give different results for different inputs
X_train = np.array([
    [1, 1, 1],  # Mild
    [5, 5, 5],  # Severe 
    [10, 10, 10],  # Moderate
    [15, 15, 15]   # Unreported
])
y_train = np.array(['Mild', 'Severe', 'Moderate', 'Unreported'])

# Fit the model
dummy_model.fit(X_train, y_train)

# Create dummy encoders and other required components
class_encoder = LabelEncoder()
class_encoder.fit(['Mild', 'Moderate', 'Severe', 'Unreported'])

# Simple dictionary-based transformer that will generate different features based on inputs
def transform_features(df):
    # Create a feature array with 3 columns
    result = np.zeros((df.shape[0], 3))
    
    # Populate based on key features
    for i in range(df.shape[0]):
        # Age factor (0-20)
        if 'Age/Years' in df.columns:
            age = df['Age/Years'].iloc[i]
            result[i, 0] = min(age, 20)
        
        # Sex factor 
        if 'Sex' in df.columns:
            sex = df['Sex'].iloc[i]
            result[i, 1] = 1 if sex == 'Male' else 5 if sex == 'Female' else 3
        
        # Route factor
        if 'Route of admin.' in df.columns:
            route = df['Route of admin.'].iloc[i]
            result[i, 2] = 1 if route == 'Intradermal' else 5 if route == 'Subcutaneous' else 10
            
    return result

dummy_data = {
    'model': dummy_model,
    'encoders': {
        'Age/Years': LabelEncoder().fit([0, 1, 2, 3]),
        'Sex': LabelEncoder().fit(['Male', 'Female', 'Unknown']),
        'Reported medication': LabelEncoder().fit(['BCG Vaccine']),
        'Indication': LabelEncoder().fit(['Immunization']),
        'Route of admin.': LabelEncoder().fit(['Intradermal', 'Subcutaneous', 'Intramuscular', 'Unknown']),
        'Time to onset': LabelEncoder().fit(['3 days']),
        'Adverse event by location': LabelEncoder().fit(['Local', 'Systemic', 'Mixed', 'Unknown']),
        'System Organ class affected': LabelEncoder().fit(['General disorders and administration site conditions']),
        'Mapped Term 1': LabelEncoder().fit(['Injection site inflammation']),
        'UMC report ID': LabelEncoder().fit([0]),
        'Country of primary source': LabelEncoder().fit(['Unknown']),
        'Outcome': LabelEncoder().fit(['Unknown']),
        'Casuality Assessment conclusion': LabelEncoder().fit(['Unknown']),
        '1st or 2nd Dose': LabelEncoder().fit(['1st dose', '2nd dose']),
        'Types of Vaccines': LabelEncoder().fit(['BCG'])
    },
    'fill_values': {
        'Age/Years': 0,
        'Sex': 'Unknown',
        'Reported medication': 'BCG Vaccine',
        'Indication': 'Immunization',
        'Route of admin.': 'Intradermal',
        'Time to onset': '3 days',
        'Adverse event by location': 'Local',
        'System Organ class affected': 'General disorders and administration site conditions',
        'Mapped Term 1': 'Injection site inflammation',
        'UMC report ID': 0,
        'Country of primary source': 'Unknown',
        'Outcome': 'Unknown',
        'Casuality Assessment conclusion': 'Unknown',
        '1st or 2nd Dose': '1st dose',
        'Types of Vaccines': 'BCG'
    },
    'scaler': None,
    'feature_selector': None,  # We'll handle transformation in utils.py
    'transform_function': transform_features,  # Store the function code
    'target_encoder': class_encoder,
    'selected_features': ['UMC report ID', 'Country of primary source', 'Age/Years', 
                         'Reported medication', 'Indication', 'Route of admin.', 'Outcome', 
                         'Casuality Assessment conclusion', '1st or 2nd Dose', 
                         'Types of Vaccines', 'Adverse event by location', 
                         'System Organ class affected', 'Mapped Term 1', 'Sex',
                         'Time to onset'],
    'is_binary': False
}

# Save the dummy model
if not os.path.exists('models'):
    os.makedirs('models')
    
with open('models/final_improved_bcg_severity_model.pkl', 'wb') as f:
    pickle.dump(dummy_data, f)

print("Simple dummy model created and saved to 'models/final_improved_bcg_severity_model.pkl'")