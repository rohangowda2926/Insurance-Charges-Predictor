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

- **Frontend**: [View Live Application](https://rohangowda2926.github.io/Insurance-Prediction-/)
- **Backend API**: Hosted on Render

## 📊 Model Performance

The model uses a **Gradient Boosting Regressor** with the following features:
- Age, BMI, Number of children
- Sex (Male/Female)
- Smoking status (Yes/No) 
- Region (Northeast, Northwest, Southeast, Southwest)

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
│       ├── model.joblib         # Trained ML model
│       ├── requirements.txt     # Python dependencies
│       ├── static/
│       │   └── app.js          # Frontend JavaScript
│       └── templates/
│           └── index.html      # Main HTML template
├── ml/
│   ├── insurance.csv           # Training dataset
│   └── train_model.py          # Model training script
├── docs/
│   ├── index.html              # GitHub Pages version
│   └── static/
│       └── app.js              # Frontend for GitHub Pages
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── extract_coefficients.py     # Model analysis utility
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
- Age: 25, BMI: 22, Non-smoker, 0 children
- **Predicted Charge**: ~$3,000-5,000

### Example 2: High Risk Profile  
- Age: 55, BMI: 35, Smoker, 3 children
- **Predicted Charge**: ~$35,000-45,000

## 🔧 Model Analysis

Use the `extract_coefficients.py` script to analyze model behavior:

```bash
python extract_coefficients.py
```

This script provides:
- Feature importance analysis
- Sample predictions
- Model structure insights

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