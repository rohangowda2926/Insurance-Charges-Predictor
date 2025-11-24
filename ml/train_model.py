import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv("insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

# Define preprocessing
categorical = ["sex", "smoker", "region"]
numeric = ["age", "bmi", "children"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

# Create pipeline
model = GradientBoostingRegressor(random_state=42)
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# Save model
joblib.dump(pipe, "../app/app/model.joblib")
print("Model retrained and saved with current scikit-learn version")