# Nigeria Agricultural Price Predictor — Streamlit App

## Project
Predictive Model of Agricultural Produce Pricing in Nigeria  
Bowen University Final Year Project

## Setup

### 1. Place these files in the same folder:
```
app.py
requirements.txt
merged_dataset_final__2_.csv      ← your dataset
model_comparison_final.json       ← your model results
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run locally
```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud (free)

1. Push all 4 files to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy** — live URL in ~2 minutes

## Features

| Tab | What it does |
|-----|-------------|
| 🔮 Price Prediction | Select year, state, market, crop → get predicted price with historical comparison |
| 📊 Data Explorer | Filter and visualise historical prices by state and crop |
| 📈 Model Performance | View MAE/RMSE/MAPE/Accuracy for all 7 models per crop, actual vs predicted chart |

## Crops covered
Yam · Gari (Yellow) · Gari (White) · Rice (Local) · Maize (White) · Maize (Yellow)

## States in dataset
Abia · Adamawa · Borno · Gombe · Jigawa · Kaduna · Kano · Katsina · Kebbi · Lagos · Oyo · Yobe · Zamfara
