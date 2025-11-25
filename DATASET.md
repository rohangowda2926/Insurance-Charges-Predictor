# Dataset Information 📊

## Overview
The insurance dataset contains **1,338 records** of medical insurance charges with 7 features. This is a regression problem where we predict continuous insurance charges based on personal and lifestyle factors.

## Dataset Source
- **Source**: Medical Cost Personal Datasets from Kaggle
- **License**: Public Domain
- **Size**: 1,338 rows × 7 columns
- **Target Variable**: `charges` (continuous)

## Features Description

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `age` | Numeric | Age of the beneficiary | 18-64 years |
| `sex` | Categorical | Gender of the beneficiary | male, female |
| `bmi` | Numeric | Body Mass Index | 15.96-53.13 |
| `children` | Numeric | Number of dependents | 0-5 |
| `smoker` | Categorical | Smoking status | yes, no |
| `region` | Categorical | Residential area | northeast, northwest, southeast, southwest |
| `charges` | Numeric | **Target**: Medical insurance charges | $1,121.87 - $63,770.43 |

## Key Statistics

### Numerical Features
- **Age**: Mean = 39.2 years, Std = 14.0 years
- **BMI**: Mean = 30.7, Std = 6.1 (indicates overweight population)
- **Children**: Mean = 1.1, Std = 1.2 (most have 0-2 children)
- **Charges**: Mean = $13,270, Std = $12,110 (high variance)

### Categorical Features
- **Sex**: 50.5% male, 49.5% female (balanced)
- **Smoker**: 20.5% smokers, 79.5% non-smokers
- **Region**: Fairly balanced across all 4 regions (~25% each)

## Data Quality
- ✅ **No missing values**
- ✅ **No duplicate records**
- ✅ **Consistent data types**
- ✅ **Realistic value ranges**

## Key Insights from EDA

### 1. Smoking Impact
- **Smokers**: Average charge = $32,050
- **Non-smokers**: Average charge = $8,434
- **Impact**: Smokers pay **3.8x more** than non-smokers

### 2. Age Correlation
- **Correlation with charges**: 0.299 (moderate positive)
- Charges generally increase with age
- Most pronounced in smokers

### 3. BMI Impact
- **Correlation with charges**: 0.198 (weak positive)
- Higher BMI associated with higher charges
- Effect amplified for smokers

### 4. Regional Differences
- **Southeast**: Highest average charges ($14,735)
- **Southwest**: Lowest average charges ($12,347)
- Difference likely due to cost of living variations

### 5. Children Impact
- **Correlation with charges**: 0.068 (very weak)
- Minimal direct impact on insurance charges
- Slight increase with more dependents

## Data Preprocessing

### Applied Transformations
1. **One-Hot Encoding**: Categorical variables (sex, smoker, region)
2. **Feature Scaling**: Not required for tree-based models
3. **Train-Test Split**: 80-20 split with random_state=42

### Final Feature Set (After Preprocessing)
- `age` (numeric)
- `bmi` (numeric) 
- `children` (numeric)
- `sex_female` (binary)
- `sex_male` (binary)
- `smoker_no` (binary)
- `smoker_yes` (binary)
- `region_northeast` (binary)
- `region_northwest` (binary)
- `region_southeast` (binary)
- `region_southwest` (binary)

**Total Features**: 11 (3 numeric + 8 binary)

## Model Suitability
This dataset is ideal for:
- **Regression modeling** (continuous target)
- **Feature importance analysis** (clear interpretable features)
- **Business insights** (healthcare cost factors)
- **Beginner ML projects** (clean, well-structured data)

## Ethical Considerations
- **Smoking status**: Legitimate risk factor for insurance
- **Age**: Standard actuarial factor
- **BMI**: Health-related risk indicator
- **Sex**: May raise fairness concerns in some jurisdictions
- **Region**: Geographic risk variation is standard practice

## Usage in Model
The dataset enables prediction of insurance charges with:
- **High accuracy** (R² = 0.88)
- **Clear feature importance** (smoking dominates)
- **Business interpretability** (actionable insights)
- **Regulatory compliance** (standard insurance factors)