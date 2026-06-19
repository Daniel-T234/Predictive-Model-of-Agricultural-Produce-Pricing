import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria Agricultural Price Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #2E7D32;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F1F8E9; border-left: 5px solid #4CAF50;
        padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0; color: #1B5E20; font-size: 0.85rem; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 1.6rem;
                      font-weight: 700; color: #2E7D32; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; background: #C8E6C9; color: #1B5E20;
    }
    .info-box {
        background: #E3F2FD; border-left: 4px solid #1976D2;
        padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.88rem; color: #0D47A1;
    }
    .warn-box {
        background: #FFF8E1; border-left: 4px solid #F9A825;
        padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.88rem; color: #6D4C41;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("merged_dataset_final__2_.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df

@st.cache_data
def load_model_results():
    with open("model_comparison_final.json") as f:
        return json.load(f)

@st.cache_resource
def train_models(df):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    crop_cols = ["yam", "gari_yellow", "gari_white", "rice_local", "maize_white", "maize_yellow"]
    le_state  = LabelEncoder()
    le_market = LabelEncoder()

    df2 = df.copy().dropna(subset=crop_cols + ["admin1", "market", "year", "month"])
    df2["state_enc"]  = le_state.fit_transform(df2["admin1"])
    df2["market_enc"] = le_market.fit_transform(df2["market"])

    feature_cols = ["year", "month", "state_enc", "market_enc",
                    "cpi", "fuel_price", "temperature", "rainfall",
                    "latitude", "longitude"]

    models = {}
    for crop in crop_cols:
        tmp = df2.dropna(subset=[crop])
        X   = tmp[feature_cols]
        y   = tmp[crop]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                      max_depth=4, random_state=42)
        m.fit(X_tr, y_tr)
        models[crop] = m

    return models, le_state, le_market, feature_cols

# ── Helpers ───────────────────────────────────────────────────────────────────
CROP_META = {
    "yam":          {"label": "Yam",           "unit": "NGN/100 kg", "emoji": "🍠"},
    "gari_yellow":  {"label": "Gari (Yellow)", "unit": "NGN/kg",     "emoji": "🌽"},
    "gari_white":   {"label": "Gari (White)",  "unit": "NGN/kg",     "emoji": "⬜"},
    "rice_local":   {"label": "Rice (Local)",  "unit": "NGN/kg",     "emoji": "🍚"},
    "maize_white":  {"label": "Maize (White)", "unit": "NGN/kg",     "emoji": "🌽"},
    "maize_yellow": {"label": "Maize (Yellow)","unit": "NGN/kg",     "emoji": "🌾"},
}

BEST_MODEL_MAP = {
    "yam":          "XGBoost (86.4%)",
    "gari_yellow":  "XGBoost (80.3%)",
    "gari_white":   "XGBoost (80.3%)",
    "rice_local":   "ARIMA/SARIMAX (80.4%)",
    "maize_white":  "XGBoost (85.5%)",
    "maize_yellow": "XGBoost (84.7%)",
}

def fmt_ngn(val):
    return f"₦{val:,.0f}"

# ── Load ──────────────────────────────────────────────────────────────────────
df          = load_data()
model_data  = load_model_results()
models, le_state, le_market, feature_cols = train_models(df)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌾 Nigeria Agricultural Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predictive Model of Agricultural Produce Pricing in Nigeria · Bowen University FYP</p>', unsafe_allow_html=True)

tab_predict, tab_explorer, tab_models = st.tabs(["🔮 Price Prediction", "📊 Data Explorer", "📈 Model Performance"])

# ════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Enter Prediction Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        year  = st.selectbox("📅 Year",  list(range(2014, 2031)),
                             index=list(range(2014, 2031)).index(2024))
        month = st.selectbox("🗓️ Month", list(range(1, 13)),
                             format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))

    with col2:
        states = sorted(df["admin1"].unique())
        state  = st.selectbox("📍 State", states)
        markets_for_state = sorted(df[df["admin1"] == state]["market"].unique())
        market = st.selectbox("🏪 Market", markets_for_state)

    with col3:
        crop_key   = st.selectbox(
            "🌱 Crop",
            list(CROP_META.keys()),
            format_func=lambda k: f"{CROP_META[k]['emoji']} {CROP_META[k]['label']}"
        )
        # Optional macro overrides
        with st.expander("⚙️ Macro Inputs (optional)", expanded=False):
            avg_cpi   = df["cpi"].mean()
            avg_fuel  = df["fuel_price"].mean()
            avg_temp  = df["temperature"].mean()
            avg_rain  = df["rainfall"].mean()
            cpi        = st.number_input("CPI",        value=float(round(avg_cpi,  1)), step=1.0)
            fuel_price = st.number_input("Fuel Price", value=float(round(avg_fuel, 1)), step=1.0)
            temperature= st.number_input("Temp (°C)",  value=float(round(avg_temp, 1)), step=0.1)
            rainfall   = st.number_input("Rainfall",   value=float(round(avg_rain, 1)), step=0.1)

    st.divider()

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):

        row = df[(df["admin1"] == state) & (df["market"] == market)]
        lat  = row["latitude"].mean()  if not row.empty else df["latitude"].mean()
        lon  = row["longitude"].mean() if not row.empty else df["longitude"].mean()

        # Encode
        state_enc = (le_state.transform([state])[0]
                     if state in le_state.classes_
                     else le_state.transform([le_state.classes_[0]])[0])
        market_enc = (le_market.transform([market])[0]
                      if market in le_market.classes_
                      else le_market.transform([le_market.classes_[0]])[0])

        X_input = pd.DataFrame([{
            "year": year, "month": month,
            "state_enc": state_enc, "market_enc": market_enc,
            "cpi": cpi, "fuel_price": fuel_price,
            "temperature": temperature, "rainfall": rainfall,
            "latitude": lat, "longitude": lon,
        }])

        pred = models[crop_key].predict(X_input)[0]
        pred = max(pred, 0)

        meta = CROP_META[crop_key]

        # Historical reference
        hist = df[(df["admin1"] == state) & (df["year"] < year)][crop_key].dropna()
        hist_mean = hist.mean() if not hist.empty else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <h3>{meta['emoji']} Predicted Price</h3>
              <p>{fmt_ngn(pred)}</p>
              <span class="badge">{meta['unit']}</span>
            </div>""", unsafe_allow_html=True)
        with c2:
            if hist_mean:
                delta = ((pred - hist_mean) / hist_mean) * 100
                direction = "📈" if delta > 0 else "📉"
                st.markdown(f"""
                <div class="metric-card">
                  <h3>vs Historical Average ({state})</h3>
                  <p>{direction} {delta:+.1f}%</p>
                  <span class="badge">Hist avg: {fmt_ngn(hist_mean)}</span>
                </div>""", unsafe_allow_html=True)
        with c3:
            best_m = BEST_MODEL_MAP.get(crop_key, "—")
            st.markdown(f"""
            <div class="metric-card">
              <h3>Best Validated Model</h3>
              <p style="font-size:1rem">{best_m}</p>
              <span class="badge">{meta['label']}</span>
            </div>""", unsafe_allow_html=True)

        # Trend chart for this crop × state
        st.markdown(f"#### 📉 Historical Trend — {meta['label']} in {state}")
        trend = (df[df["admin1"] == state]
                 .groupby("year")[crop_key].mean()
                 .dropna()
                 .reset_index())
        if not trend.empty:
            new_row = pd.DataFrame({"year": [year], crop_key: [pred]})
            trend   = pd.concat([trend, new_row], ignore_index=True)
            trend   = trend.rename(columns={crop_key: "Price (NGN)"})
            st.line_chart(trend.set_index("year"), use_container_width=True)

        st.markdown(f"""<div class="warn-box">
        ⚠️ This prediction is generated by a Gradient Boosting model trained on historical NBS/WFP
        market data (2009–2023). Predictions beyond the training window carry higher uncertainty.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 – DATA EXPLORER
# ════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### 📊 Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_state = st.selectbox("State", ["All"] + sorted(df["admin1"].unique()), key="exp_state")
    with col2:
        sel_crop  = st.selectbox("Crop", list(CROP_META.keys()),
                                  format_func=lambda k: CROP_META[k]["label"], key="exp_crop")
    with col3:
        yr_range = st.slider("Year range", int(df["year"].min()), int(df["year"].max()),
                             (2015, 2023), key="yr_range")

    filt = df[(df["year"] >= yr_range[0]) & (df["year"] <= yr_range[1])]
    if sel_state != "All":
        filt = filt[filt["admin1"] == sel_state]

    agg = filt.groupby(["year", "admin1"])[sel_crop].mean().reset_index()
    agg.columns = ["Year", "State", "Avg Price"]

    st.markdown(f"#### Average {CROP_META[sel_crop]['label']} Price by Year")
    pivot = agg.pivot(index="Year", columns="State", values="Avg Price")
    st.line_chart(pivot, use_container_width=True)

    st.markdown("#### State-level Summary Table")
    summary = (filt.groupby("admin1")[sel_crop]
               .agg(["mean", "min", "max", "count"])
               .rename(columns={"mean":"Avg Price","min":"Min","max":"Max","count":"Records"})
               .reset_index()
               .rename(columns={"admin1":"State"}))
    for col in ["Avg Price","Min","Max"]:
        summary[col] = summary[col].apply(lambda x: f"₦{x:,.0f}" if pd.notna(x) else "—")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### Raw Data Sample")
    st.dataframe(filt[["date","admin1","market",sel_crop,"cpi","fuel_price"]].head(50),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### 📈 Model Comparison Results")

    sel_crop_m = st.selectbox(
        "Select Crop", list(model_data.keys()),
        format_func=lambda k: model_data[k]["label"], key="mod_crop"
    )

    cdata = model_data[sel_crop_m]
    model_keys = ["arima","xgboost","random_forest","hybrid_arima_xgb",
                  "hybrid_arima_rf","hybrid_xgb_rf","hybrid_triple"]
    model_labels = {
        "arima":"ARIMA","xgboost":"XGBoost","random_forest":"Random Forest",
        "hybrid_arima_xgb":"Hybrid ARIMA+XGB","hybrid_arima_rf":"Hybrid ARIMA+RF",
        "hybrid_xgb_rf":"Hybrid XGB+RF","hybrid_triple":"Hybrid Triple"
    }

    rows = []
    for k in model_keys:
        m = cdata[k]
        rows.append({
            "Model":    m["model"],
            "MAE":      m["MAE"],
            "RMSE":     m["RMSE"],
            "R²":       m["R2"],
            "MAPE (%)": m["MAPE"],
            "Accuracy": m["accuracy"],
        })

    perf_df = pd.DataFrame(rows)

    # Highlight best
    best = cdata["best_model"]
    st.markdown(f"**🏆 Best Model for {cdata['label']}:** `{best}` — "
                f"Accuracy: `{cdata['best_metrics']['accuracy']}%` | "
                f"MAE: `{cdata['best_metrics']['MAE']}` | "
                f"MAPE: `{cdata['best_metrics']['MAPE']}%`")
    st.divider()

    # Bar chart: accuracy
    acc_df = perf_df[["Model","Accuracy"]].set_index("Model")
    st.markdown("#### Model Accuracy (%)")
    st.bar_chart(acc_df, use_container_width=True)

    # Metrics table
    st.markdown("#### Full Metrics Table")
    st.dataframe(
        perf_df.style.highlight_max(subset=["Accuracy"], color="#C8E6C9")
                     .highlight_min(subset=["MAE","RMSE","MAPE (%)"], color="#C8E6C9"),
        use_container_width=True, hide_index=True
    )

    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted (Test Period)")
    dates = pd.to_datetime(cdata["test_dates"])

    chart_df = pd.DataFrame({
        "Actual":         cdata["actual"],
        "XGBoost":        cdata.get("xgb_pred", []),
        "Random Forest":  cdata.get("rf_pred", []),
        "ARIMA":          cdata.get("arima_pred", []),
    }, index=dates)
    st.line_chart(chart_df, use_container_width=True)

    st.markdown(f"""<div class="info-box">
    Unit: <strong>{cdata['unit']}</strong> · Test period: Feb 2023 – Jan 2024.
    Models evaluated on held-out data. Best model selected by lowest MAPE.
    </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<center style="font-size:0.8rem; color:#888;">
🌾 Bowen University FYP · Predictive Model of Agricultural Produce Pricing in Nigeria<br>
Data: NBS / WFP Market Price Monitoring · 2009–2023
</center>
""", unsafe_allow_html=True)
