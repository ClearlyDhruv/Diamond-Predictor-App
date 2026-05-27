import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.request
import json
from price_predictor import train_price_predictor, predict_price

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 30px 20px;
        border-radius: 15px;
        margin-bottom: 25px;
    }
    .main-header h1 { font-size: 2.8em; margin: 0; letter-spacing: 2px; }
    .main-header p  { font-size: 1.1em; margin: 8px 0 0 0; opacity: 0.85; }

    .price-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 35px;
        border-radius: 18px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .price-result .label     { font-size: 1em; opacity: 0.9; letter-spacing: 1px; text-transform: uppercase; }
    .price-result .amount    { font-size: 3.2em; font-weight: 800; margin: 5px 0; }
    .price-result .amount-secondary { font-size: 1.6em; font-weight: 600; opacity: 0.85; margin: 0 0 8px 0; }
    .price-result .sub       { font-size: 0.95em; opacity: 0.8; }

    .info-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px 18px;
        margin: 8px 0;
    }
    .info-card .card-label { font-size: 0.78em; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .info-card .card-value { font-size: 1.4em; font-weight: 700; color: #2c3e50; margin-top: 2px; }

    .section-header {
        font-size: 1.05em;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #667eea;
        padding-bottom: 6px;
        margin-bottom: 15px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px;
        font-size: 1.1em;
        font-weight: 700;
        border-radius: 10px;
        letter-spacing: 1px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    .sidebar-stat {
        background: #f0f2ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.92em;
        color: #1a1a2e;
    }
    .sidebar-stat b { color: #4a4ebd; }

    .rate-badge {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.82em;
        color: #2e7d32;
        text-align: center;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Exchange rate (try live, fall back to hardcoded) ───────────────────────────
@st.cache_data(ttl=3600)   # refresh once per hour
def get_usd_to_inr():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.loads(r.read())
        rate = data["rates"]["INR"]
        return rate, "live"
    except Exception:
        return 84.5, "fixed fallback"   # update this if it drifts a lot


USD_TO_INR, rate_source = get_usd_to_inr()


def fmt_inr(amount_usd):
    """Format USD amount as Indian Rupees with lakh/crore formatting."""
    inr = amount_usd * USD_TO_INR
    # Indian number system: xx,xx,xxx
    s = f"{inr:,.0f}"
    # Re-format with Indian grouping
    parts = s.replace(",", "")
    if len(parts) > 3:
        last3 = parts[-3:]
        rest   = parts[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        groups.reverse()
        s = ",".join(groups) + "," + last3
    return f"₹{s}"


# ── Load & train (cached so it only runs once per session) ─────────────────────
@st.cache_resource(show_spinner="⏳ Training model on 53,940 diamonds — takes ~15 seconds…")
def load_model():
    csv_path = os.path.join(os.path.dirname(__file__), "diamonds.csv")
    df = pd.read_csv(csv_path)
    metrics = train_price_predictor(df)
    return df, metrics

df, metrics = load_model()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>💎 Diamond Price Predictor</h1>
  <p>Machine Learning · XGBoost · R² = {r2:.4f} · Trained on {n:,} diamonds</p>
</div>
""".format(r2=metrics["r2"], n=metrics["n_train"]), unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    st.markdown(f'<div class="sidebar-stat"><b>R² Score</b> — {metrics["r2"]:.4f} ({metrics["r2"]*100:.1f}% variance explained)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat"><b>Mean Absolute Error</b> — ${metrics["mae"]:,.0f} / {fmt_inr(metrics["mae"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat"><b>RMSE</b> — ${metrics["rmse"]:,.0f} / {fmt_inr(metrics["rmse"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat"><b>Training samples</b> — {metrics["n_train"]:,}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💱 Exchange Rate")
    st.markdown(f'<div class="sidebar-stat"><b>1 USD</b> = ₹{USD_TO_INR:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rate-badge">{"🟢 Live rate" if rate_source == "live" else "🟡 Fallback rate (offline)"}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📈 Dataset Stats")
    feat = st.selectbox("Explore feature:", ["carat", "depth", "table", "x", "y", "z"])
    col = df[feat]
    st.markdown(f'<div class="sidebar-stat"><b>Mean</b> — {col.mean():.3f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat"><b>Median</b> — {col.median():.3f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-stat"><b>Min / Max</b> — {col.min():.2f} / {col.max():.2f}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎓 About")
    st.caption("Diamond Price Predictor · BE (CSE) Final Project · Chitkara University · Dhruv Nagpal (2211981142)")


# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">✨ Quality Characteristics (4Cs)</div>', unsafe_allow_html=True)

    carat = st.slider("Carat  (weight)", 0.20, 5.00, 1.00, 0.01,
                      help="Weight of the diamond. Strongest price driver.")

    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"],
                       index=4, help="Ideal > Premium > Very Good > Good > Fair")

    color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"],
                         index=3, help="D (colourless, best) → J (noticeable tint)")

    clarity = st.selectbox("Clarity",
                           ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
                           index=4, help="IF (flawless) → I1 (inclusions visible)")

with col2:
    st.markdown('<div class="section-header">📏 Physical Dimensions</div>', unsafe_allow_html=True)

    depth = st.slider("Depth %", 43.0, 79.0, 61.7, 0.1,
                      help="Total depth as % of average diameter")
    table = st.slider("Table %", 43.0, 95.0, 57.0, 0.1,
                      help="Width of top facet relative to widest point")

    dim_c1, dim_c2, dim_c3 = st.columns(3)
    with dim_c1:
        x = st.number_input("Length X (mm)", 3.0, 11.0, 5.50, 0.01)
    with dim_c2:
        y = st.number_input("Width Y (mm)",  3.0, 11.0, 5.51, 0.01)
    with dim_c3:
        z = st.number_input("Depth Z (mm)",  2.0,  6.5, 3.40, 0.01)

    st.markdown("")
    st.markdown("")
    predict_clicked = st.button("🔮  PREDICT PRICE", use_container_width=True)


# ── Result ─────────────────────────────────────────────────────────────────────
st.markdown("---")

if predict_clicked:
    features = {
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table, "x": x, "y": y, "z": z,
    }

    try:
        price_usd = predict_price(features)
        price_inr = price_usd * USD_TO_INR

        # Main price card — shows BOTH currencies
        st.markdown(f"""
        <div class="price-result">
          <div class="label">💰 Estimated Market Price</div>
          <div class="amount">{fmt_inr(price_usd)}</div>
          <div class="amount-secondary">${price_usd:,.0f} USD</div>
          <div class="sub">XGBoost Prediction · ±{fmt_inr(metrics['mae'])} typical error · 1 USD = ₹{USD_TO_INR:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Supporting metrics
        m1, m2, m3, m4 = st.columns(4)

        price_per_carat_inr = price_inr / carat
        size_label = "Tiny" if carat < 0.3 else "Small" if carat < 0.7 else "Medium" if carat < 1.5 else "Large" if carat < 3 else "Exceptional"

        quality = sum([
            2 if cut in ["Premium", "Ideal"] else 1,
            2 if color in ["D", "E", "F"] else (1 if color in ["G", "H"] else 0),
            2 if clarity in ["VVS1", "VVS2", "IF"] else (1 if clarity in ["VS1", "VS2"] else 0),
        ])
        stars = "⭐" * min(quality, 5) + "☆" * max(0, 5 - quality)

        with m1:
            st.markdown(f'<div class="info-card"><div class="card-label">Price / Carat</div><div class="card-value">{fmt_inr(price_usd / carat)}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="info-card"><div class="card-label">Size Category</div><div class="card-value">{size_label}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="info-card"><div class="card-label">Quality Rating</div><div class="card-value">{stars}</div></div>', unsafe_allow_html=True)
        with m4:
            volume = x * y * z
            st.markdown(f'<div class="info-card"><div class="card-label">Volume (mm³)</div><div class="card-value">{volume:.2f}</div></div>', unsafe_allow_html=True)

        # Input summary table
        st.markdown("")
        with st.expander("📋 View full input summary"):
            summary = pd.DataFrame({
                "Feature": ["Carat", "Cut", "Color", "Clarity", "Depth %", "Table %", "X (mm)", "Y (mm)", "Z (mm)"],
                "Value":   [carat, cut, color, clarity, f"{depth:.1f}%", f"{table:.1f}%", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"],
            })
            st.table(summary)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👆 Set the diamond characteristics above and click **PREDICT PRICE**.")

    st.markdown("#### 📦 Sample diamonds from the dataset")
    sample = df[["carat", "cut", "color", "clarity", "depth", "table", "price"]].sample(6, random_state=7).copy()
    sample["price (USD)"] = sample["price"].apply(lambda p: f"${p:,.0f}")
    sample["price (INR)"] = sample["price"].apply(lambda p: fmt_inr(p))
    sample = sample.drop(columns=["price"])
    st.dataframe(sample.reset_index(drop=True), use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.85em'>"
    "💎 Diamond Price Predictor · XGBoost · scikit-learn · Streamlit · "
    "Dhruv Nagpal & Spathneja · Chitkara University 2026"
    "</div>",
    unsafe_allow_html=True,
)
