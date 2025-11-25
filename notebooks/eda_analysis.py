import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def perform_eda():
    """Comprehensive Exploratory Data Analysis"""
    
    # Load data
    df = pd.read_csv('../ml/insurance.csv')
    
    print("📊 INSURANCE DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic info
    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    
    # Statistical summary
    print("\n📈 STATISTICAL SUMMARY")
    print("=" * 30)
    print(df.describe())
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Distribution of charges
    plt.subplot(3, 4, 1)
    plt.hist(df['charges'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges ($)')
    plt.ylabel('Frequency')
    
    # 2. Charges by smoker status
    plt.subplot(3, 4, 2)
    sns.boxplot(data=df, x='smoker', y='charges', palette='Set2')
    plt.title('Charges by Smoking Status')
    plt.ylabel('Charges ($)')
    
    # 3. Charges by sex
    plt.subplot(3, 4, 3)
    sns.boxplot(data=df, x='sex', y='charges', palette='Set1')
    plt.title('Charges by Gender')
    plt.ylabel('Charges ($)')
    
    # 4. Charges by region
    plt.subplot(3, 4, 4)
    sns.boxplot(data=df, x='region', y='charges', palette='viridis')
    plt.title('Charges by Region')
    plt.xticks(rotation=45)
    plt.ylabel('Charges ($)')
    
    # 5. Age vs Charges
    plt.subplot(3, 4, 5)
    plt.scatter(df['age'], df['charges'], alpha=0.6, color='coral')
    plt.title('Age vs Charges')
    plt.xlabel('Age')
    plt.ylabel('Charges ($)')
    
    # 6. BMI vs Charges
    plt.subplot(3, 4, 6)
    plt.scatter(df['bmi'], df['charges'], alpha=0.6, color='lightgreen')
    plt.title('BMI vs Charges')
    plt.xlabel('BMI')
    plt.ylabel('Charges ($)')
    
    # 7. Children vs Charges
    plt.subplot(3, 4, 7)
    sns.boxplot(data=df, x='children', y='charges', palette='pastel')
    plt.title('Charges by Number of Children')
    plt.ylabel('Charges ($)')
    
    # 8. Correlation heatmap
    plt.subplot(3, 4, 8)
    # Encode categorical variables for correlation
    df_encoded = pd.get_dummies(df)
    correlation_matrix = df_encoded.corr()
    sns.heatmap(correlation_matrix[['charges']].sort_values('charges', ascending=False), 
                annot=True, cmap='RdYlBu_r', center=0)
    plt.title('Feature Correlation with Charges')
    
    # 9. Age distribution
    plt.subplot(3, 4, 9)
    plt.hist(df['age'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # 10. BMI distribution
    plt.subplot(3, 4, 10)
    plt.hist(df['bmi'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    
    # 11. Smoker vs Non-smoker charges (detailed)
    plt.subplot(3, 4, 11)
    smoker_charges = df[df['smoker'] == 'yes']['charges']
    non_smoker_charges = df[df['smoker'] == 'no']['charges']
    plt.hist([non_smoker_charges, smoker_charges], bins=30, alpha=0.7, 
             label=['Non-smoker', 'Smoker'], color=['lightblue', 'salmon'])
    plt.title('Charges Distribution: Smoker vs Non-smoker')
    plt.xlabel('Charges ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 12. Regional analysis
    plt.subplot(3, 4, 12)
    region_stats = df.groupby('region')['charges'].mean().sort_values(ascending=False)
    plt.bar(region_stats.index, region_stats.values, color='teal', alpha=0.7)
    plt.title('Average Charges by Region')
    plt.xlabel('Region')
    plt.ylabel('Average Charges ($)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../docs/eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Key insights
    print("\n🔍 KEY INSIGHTS")
    print("=" * 20)
    
    smoker_avg = df[df['smoker'] == 'yes']['charges'].mean()
    non_smoker_avg = df[df['smoker'] == 'no']['charges'].mean()
    print(f"Average charges for smokers: ${smoker_avg:,.2f}")
    print(f"Average charges for non-smokers: ${non_smoker_avg:,.2f}")
    print(f"Smokers pay {smoker_avg/non_smoker_avg:.1f}x more than non-smokers")
    
    age_corr = df['age'].corr(df['charges'])
    bmi_corr = df['bmi'].corr(df['charges'])
    print(f"\nAge correlation with charges: {age_corr:.3f}")
    print(f"BMI correlation with charges: {bmi_corr:.3f}")
    
    print(f"\nAge range: {df['age'].min()} - {df['age'].max()} years")
    print(f"BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    print(f"Charges range: ${df['charges'].min():,.2f} - ${df['charges'].max():,.2f}")
    
    return df

if __name__ == "__main__":
    data = perform_eda()