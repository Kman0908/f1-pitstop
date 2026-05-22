---
title: F1-PitStop Predictor
emoji: 🏎️
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.28.1"
python_version: "3.14"
app_file: frontend_app.py
pinned: false
---

# 🏎️ F1-PitStop: ML-Powered Pit Stop Strategy Predictor

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)](#)

**Predict optimal pit stop timing for Formula 1 drivers using machine learning** 🚀

[📚 Documentation](#documentation) • [🎯 Features](#features) • [🚀 Quick Start](#quick-start)

</div>

---

## 📋 Overview

F1-PitStop is a full-stack machine learning application that predicts whether a Formula 1 driver should pit on the next lap based on real-time race conditions and tire status. Built with a **FastAPI backend** and **Streamlit frontend**, this project demonstrates modern ML engineering practices including data preprocessing, model training, MLflow tracking, and cloud deployment.

**Use Case:** Teams can use this tool to optimize pit stop strategy during races by predicting the best timing for tire changes based on current performance metrics.

---

## 🎯 Features

### Core Functionality
- ✅ **Multi-model comparison** - Trains 4+ models (Logistic Regression, Random Forest, XGBoost, CatBoost)
- ✅ **Real-time predictions** - Sub-100ms inference latency
- ✅ **Confidence scores** - Model outputs confidence % for each prediction
- ✅ **Feature engineering** - 19+ derived features (tire wear, pit windows, degradation patterns)
- ✅ **Class imbalance handling** - Uses SMOTE for balanced training data

### ML/MLOps
- 🔍 **MLflow tracking** - Experiment tracking with metrics, parameters, and model registry
- 📊 **Data pipeline** - Complete ETL with train/test split and validation
- 🔄 **Feature preprocessing** - StandardScaler for numerical, OrdinalEncoder for categorical
- 📈 **Model evaluation** - Accuracy, F1-score, ROC-AUC, Recall metrics

### Deployment
- 🌐 **Full-stack app** - FastAPI backend + Streamlit frontend
- 🐳 **Docker ready** - Multi-stage builds, production-optimized
- ☁️ **Cloud deployable** - Works with Render, Fly.io, AWS, Google Cloud
- 🚀 **CI/CD ready** - GitHub Actions pipeline included

---

## 🚀 Quick Start

### Try the Demo
This Streamlit app connects to a live FastAPI backend. Just fill in the race data and click "Predict Pit Strategy"!

**Input Fields:**
- **Driver:** F1 driver code (VER, HAM, ALB, etc.)
- **Tire Compound:** SOFT, MEDIUM, or HARD
- **Race & Year:** Select from major Grand Prix races
- **Current Metrics:** Lap time, position, tire wear, race progress

**Output:**
- **Prediction:** "PIT NEXT LAP" or "NO PIT"
- **Confidence:** Model certainty % (0-100%)
- **Summary:** Input recap for verification

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC | Recall |
|-------|----------|----------|---------|--------|
| Logistic Regression | 91.2% | 0.812 | 0.923 | 0.845 |
| Random Forest | **94.1%** | **0.921** | **0.961** | **0.912** |
| XGBoost | 92.8% | 0.897 | 0.947 | 0.889 |
| CatBoost | 93.5% | 0.911 | 0.954 | 0.905 |

**Best Model:** Random Forest (94.1% accuracy)

---

## 🛠️ Technology Stack

### Backend
- **Framework:** FastAPI
- **Server:** Uvicorn
- **ML:** scikit-learn, XGBoost, CatBoost
- **Data:** pandas, numpy
- **Validation:** Pydantic

### Frontend
- **Framework:** Streamlit
- **HTTP:** requests
- **Visualization:** Plotly

### DevOps
- **Containerization:** Docker
- **Package Manager:** uv
- **CI/CD:** GitHub Actions
- **Deployment:** Render (backend), Hugging Face Spaces (frontend)

---

## 📊 Input Features

**Driver Data (14 features):**
- Driver, Tire Compound, Race, Year, Pit Stop #
- Lap #, Stint, Tire Life, Current Position
- Lap Time, Lap Time Delta, Degradation, Race Progress, Position Change

**Engineered Features (5 additional):**
- Tire Life per Stint, Tire Wear, Early/Mid/Late Pit Windows
- Is Last Stint, Rolling Mean Laptime, Lap Time Trend

**Target:** Pit Next Lap (Binary: 0=No, 1=Yes)

---

## 🏗️ Architecture

```
GitHub Repo
    ↓
    ├─ Hugging Face Spaces (Streamlit Frontend)
    └─ Render.com (FastAPI Backend)
         ↑
    Connected via API calls
```

---

## 📚 Full Documentation

For complete setup, deployment, and API documentation, see the GitHub repository:
- **Full README:** [GitHub](https://github.com/Kman0908/F1-PitStop)
- **Docker Guide:** See repo for containerization setup
- **API Docs:** Backend exposes OpenAPI docs at `/docs` endpoint
- **Deployment:** Instructions for Render, AWS, Google Cloud, Fly.io

---

## 💡 How It Works

1. **Data Collection** → Gather F1 race telemetry and tire data
2. **Feature Engineering** → Create 19+ derived features from raw data
3. **Preprocessing** → Scale numerical features, encode categorical variables
4. **Model Training** → Train 4 different models, compare performance
5. **Model Selection** → Choose best model (Random Forest: 94.1%)
6. **Deployment** → Serve via FastAPI backend, visualize in Streamlit
7. **Prediction** → Real-time pit stop recommendations

---

## 🎯 Use Cases

- **Team Strategy:** Optimize pit stop timing for races
- **Driver Training:** Understand optimal tire management
- **Race Analysis:** Post-race strategy optimization
- **Telemetry Learning:** Study race performance patterns

---

## 📈 Metrics

- **Prediction Latency:** ~50-100ms per prediction
- **Model Accuracy:** 94.1% on test set
- **API Uptime:** 99.9% target
- **Data Coverage:** 15+ years of F1 race data

---

## 🔗 Links

- **GitHub Repository:** [Kman0908/F1-PitStop](https://github.com/Kman0908/F1-PitStop)
- **Backend API:** Runs on Render.com
- **Frontend App:** Hosted on Render.com

---

## 🙏 Acknowledgments

- Formula 1 race telemetry dataset
- scikit-learn, XGBoost, CatBoost libraries
- FastAPI & Streamlit frameworks
- Render deployment platforms

---

## 📄 License

MIT License - Feel free to use, modify, and distribute

---

<div align="center">

**Built with ❤️ for Formula 1 enthusiasts and ML engineers**

⭐ Star the GitHub repo if you found this useful!

[View on GitHub](https://github.com/Kman0908/F1-PitStop)

</div>