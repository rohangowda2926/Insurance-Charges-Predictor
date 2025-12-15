import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def evaluate_model():
    """Comprehensive model evaluation with metrics and visualizations"""
    
    # Load data and model
    df = pd.read_csv('../ml/insurance.csv')
    model = joblib.load('../app/app/model.joblib')
    
    # Prepare data
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Print metrics
    print("🎯 MODEL PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Testing R²:    {test_r2:.4f}")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Testing RMSE:  ${test_rmse:,.2f}")
    print(f"Training MAE:  ${train_mae:,.2f}")
    print(f"Testing MAE:   ${test_mae:,.2f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Insurance Charges Prediction - Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='steelblue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Charges ($)')
    axes[0, 0].set_ylabel('Predicted Charges ($)')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Charges ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature importance
    feature_names = ['sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 
                    'region_northeast', 'region_northwest', 'region_southeast', 
                    'region_southwest', 'age', 'bmi', 'children']
    
    # Get feature importance from the model
    importance = model.named_steps['model'].feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    axes[1, 0].bar(range(len(importance)), importance[indices], color='lightcoral')
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].set_xticks(range(len(importance)))
    axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    # 4. Distribution of predictions
    axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='Actual', color='skyblue')
    axes[1, 1].hist(y_test_pred, bins=30, alpha=0.7, label='Predicted', color='lightgreen')
    axes[1, 1].set_xlabel('Charges ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution: Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../docs/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance analysis
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    for i in indices[:5]:
        print(f"{feature_names[i]:15}: {importance[i]:.4f}")
    
    return {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'feature_importance': dict(zip(feature_names, importance))
    }

if __name__ == "__main__":
    metrics = evaluate_model()
