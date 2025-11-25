# Insurance Charges Predictor 🏥💰

A machine learning web application that predicts yearly medical insurance charges based on personal and lifestyle factors. Built with FastAPI backend and modern vanilla JavaScript frontend.

## 🌟 Features

- **Real-time Predictions**: Get instant insurance charge estimates
- **Interactive Web Interface**: Clean, responsive UI with dark theme
- **Machine Learning Model**: Gradient Boosting Regressor trained on insurance data
- **Risk Assessment**: Visual risk categorization (Low/Medium/High)
- **Profile Insights**: Dynamic analysis of key risk factors
- **GitHub Pages Deployment**: Automated deployment pipeline

## 🚀 Live Demo

- **Frontend**: [View Live Application](https://insurance-prediction-paiv.onrender.com/)
- **Backend API**: Hosted on Render

## 📊 Model Performance

### Algorithm: Gradient Boosting Regressor

**Key Metrics:**
- **R² Score**: 0.8795 (87.95% variance explained)
- **RMSE**: $4,325.87 (Root Mean Square Error)
- **MAE**: $2,402.02 (Mean Absolute Error)

**Features Used:**
- **Age** (18-64 years)
- **BMI** (Body Mass Index: 15.96-53.13)
- **Children** (Number of dependents: 0-5)
- **Sex** (Male/Female)
- **Smoking Status** (Yes/No) - *Most important feature*
- **Region** (Northeast, Northwest, Southeast, Southwest)

### Feature Importance Analysis
1. **Smoking Status**: 67.65% (smoker_yes: 46.11% + smoker_no: 21.54%)
2. **BMI**: 19.07%
3. **Age**: 11.70%
4. **Children**: 1.01%
5. **Region & Sex**: <1% each

> **Key Finding**: Smokers pay ~3.2x more than non-smokers on average

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **Uvicorn** - ASGI server

### Frontend
- **Vanilla JavaScript** - No frameworks, pure JS
- **HTML5 & CSS3** - Modern responsive design
- **Google Fonts** - Inter font family

### DevOps
- **GitHub Actions** - CI/CD pipeline
- **GitHub Pages** - Static site hosting
- **Render** - Backend API hosting

## 📁 Project Structure

```
Insurance-Prediction-/
├── app/
│   └── app/
│       ├── main.py              # FastAPI application
│       ├── model.joblib         # Trained ML model (R²=0.88)
│       ├── requirements.txt     # Python dependencies
│       ├── static/
│       │   └── app.js          # Frontend JavaScript
│       └── templates/
│           └── index.html      # Main HTML template
├── ml/
│   ├── insurance.csv           # Training dataset (1,338 records)
│   └── train_model.py          # Model training script
├── notebooks/
│   ├── model_evaluation.py     # Model metrics & visualizations
│   └── eda_analysis.py         # Exploratory data analysis
├── docs/
│   ├── index.html              # GitHub Pages version
│   ├── model_evaluation.png    # Model performance plots
│   ├── eda_analysis.png        # EDA visualizations
│   └── static/
│       └── app.js              # Frontend for GitHub Pages
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── extract_coefficients.py     # Model analysis utility
├── requirements.txt            # Root dependencies
├── runtime.txt                 # Python version for deployment
└── README.md
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohangowda2926/Insurance-Prediction-.git
   cd Insurance-Prediction-
   ```

2. **Install dependencies**
   ```bash
   cd app/app
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

4. **Open your browser**
   ```
   http://localhost:8000
   ```

### Training the Model

1. **Navigate to ML directory**
   ```bash
   cd ml/
   ```

2. **Run training script**
   ```bash
   python train_model.py
   ```

### Model Evaluation & Analysis

1. **Install analysis dependencies**
   ```bash
   cd notebooks/
   pip install -r requirements.txt
   ```

2. **Run model evaluation**
   ```bash
   python model_evaluation.py
   ```

3. **Run EDA analysis**
   ```bash
   python eda_analysis.py
   ```

## 📋 API Documentation

### Endpoints

#### `GET /`
Serves the main web interface

#### `POST /predict`
Predicts insurance charges based on input features

**Request Body:**
```json
{
  "age": 30,
  "sex": "male",
  "bmi": 25.0,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```

**Response:**
```json
{
  "predicted_charge": 4500.25
}
```

## 🎯 Usage Examples

### Example 1: Low Risk Profile
- **Input**: Age: 25, BMI: 22, Non-smoker, 0 children, Female, Southeast
- **Predicted Charge**: ~$3,282
- **Risk Level**: Low

### Example 2: High Risk Profile  
- **Input**: Age: 55, BMI: 35, Smoker, 3 children, Male, Northwest
- **Predicted Charge**: ~$39,871
- **Risk Level**: High

### Example 3: Moderate Risk Profile
- **Input**: Age: 35, BMI: 28, Non-smoker, 2 children, Male, Northeast
- **Predicted Charge**: ~$8,547
- **Risk Level**: Moderate

## 🔧 Model Analysis & Evaluation

### Run Model Evaluation
```bash
cd notebooks/
python model_evaluation.py
```

**Generates:**
- Performance metrics (R², RMSE, MAE)
- Actual vs Predicted plots
- Residuals analysis
- Feature importance visualization

### Run EDA Analysis
```bash
cd notebooks/
python eda_analysis.py
```

**Generates:**
- Dataset statistics and insights
- Distribution plots
- Correlation analysis
- Smoker vs non-smoker analysis

### Extract Model Coefficients
```bash
python extract_coefficients.py
```

**Provides:**
- Feature importance analysis
- Sample predictions
- Client-side model coefficients

## 🚀 Deployment

### Frontend Deployment (GitHub Pages)
**Note**: GitHub Pages requires a public repository for free accounts.

To enable GitHub Pages:
1. Make repository public in Settings > Danger Zone
2. Enable Pages in Settings > Pages > Source: GitHub Actions
3. Push changes to trigger deployment

### Backend Deployment (Render)
The FastAPI backend is deployed on Render:

1. **Connect Repository**: Link your GitHub repo to Render
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment**: Python 3.8+
5. **Auto-Deploy**: Enabled for main branch

#### Render Configuration
- **Service Type**: Web Service
- **Runtime**: Python 3
- **Build Command**: `cd app/app && pip install -r requirements.txt`
- **Start Command**: `cd app/app && uvicorn main:app --host 0.0.0.0 --port $PORT`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: Medical Cost Personal Datasets from Kaggle
- Icons: Emoji icons for visual elements
- Fonts: Google Fonts (Inter)

## 📞 Contact

**Rohan Dharmendra**
- GitHub: [@rohangowda2926](https://github.com/rohangowda2926)
- Project Link: [https://github.com/rohangowda2926/Insurance-Prediction-](https://github.com/rohangowda2926/Insurance-Prediction-)

---

⭐ **Star this repository if you found it helpful!**
