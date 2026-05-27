# 💎 Diamond Price Predictor

Machine Learning based diamond price estimation · XGBoost · Streamlit · Chitkara University

---

## Run Locally (Recommended for demo day)

**Windows:** Double-click `run.bat`

**Mac / Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Manual:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501** automatically.

---

## Free Online Hosting (Streamlit Community Cloud)

1. Make sure `diamonds.csv` is in the root of this repo on GitHub ✓
2. Go to **https://share.streamlit.io** and sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - Repository: `Spathneja21/Diamond_price_predictor`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. Click **Deploy** — takes ~3 minutes
6. You get a free public URL to share 🎉

---

## Project Structure

```
├── streamlit_app.py      ← Main UI (run this)
├── price_predictor.py    ← ML model (XGBoost training + prediction)
├── diamonds.csv          ← Dataset (53,940 records from Kaggle)
├── requirements.txt      ← Python dependencies
├── run.bat               ← One-click launcher (Windows)
├── run.sh                ← One-click launcher (Mac/Linux)
└── .gitignore
```

---

## Model Details

| Model    | R² Score | MAE      |
|----------|----------|----------|
| XGBoost  | ~0.98    | ~$196    |

Features used: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`
