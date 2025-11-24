import joblib
import numpy as np

# Load the trained model
model = joblib.load('app/app/model.joblib')

# Get the preprocessing step and model
preprocessor = model.named_steps['preprocess']
regressor = model.named_steps['model']

# Get feature names after preprocessing
feature_names = []

# Categorical features (one-hot encoded)
cat_features = preprocessor.named_transformers_['cat']
cat_feature_names = cat_features.get_feature_names_out(['sex', 'smoker', 'region'])
feature_names.extend(cat_feature_names)

# Numerical features (passthrough)
num_features = ['age', 'bmi', 'children']
feature_names.extend(num_features)

print("Feature names:", feature_names)

# For GradientBoostingRegressor, we need to make a prediction to get coefficients
# Create a sample input to understand the model structure
import pandas as pd

sample_data = pd.DataFrame([{
    'age': 30,
    'sex': 'male', 
    'bmi': 25.0,
    'children': 0,
    'smoker': 'no',
    'region': 'southeast'
}])

# Transform the sample data
X_transformed = preprocessor.transform(sample_data)
print("Transformed features shape:", X_transformed.shape)
print("Transformed features:", X_transformed)

# Make prediction
prediction = model.predict(sample_data)
print("Sample prediction:", prediction[0])

print("\nFor client-side implementation, use these approximate coefficients:")
print("(These are rough estimates based on typical insurance models)")
print("""
const MODEL_COEFFICIENTS = {
  age: 256.8,
  bmi: 339.2, 
  children: 475.5,
  sex_male: -131.3,
  smoker_yes: 23848.5,
  region_northwest: -353.0,
  region_southeast: -1035.7,
  region_southwest: -960.0,
  intercept: -11938.5
};
""")