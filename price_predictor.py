import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Global variables — shared across the app session
model = None
scaler = None
label_encoders = {}
feature_columns = None
numeric_feature_cols = None


def train_price_predictor(df):
    global model, scaler, label_encoders, feature_columns, numeric_feature_cols

    df = df.copy()
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Replace invalid zeros with NaN then impute
    df[numeric_cols] = df[numeric_cols].replace(0, np.nan)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals and store encoders for prediction time
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Remove outliers (IQR) on numeric columns
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Split BEFORE scaling so price stays in original USD
    X = df.drop("price", axis=1)
    y = df["price"]

    feature_columns = X.columns.tolist()
    numeric_feature_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    X = X.copy()
    X[numeric_feature_cols] = scaler.fit_transform(X[numeric_feature_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {"r2": r2, "mae": mae, "rmse": rmse, "n_train": len(X_train)}


def predict_price(features_dict):
    global model, scaler, label_encoders, feature_columns, numeric_feature_cols

    if model is None:
        raise ValueError("Model not trained yet.")

    df_input = pd.DataFrame([features_dict])

    for col in df_input.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col])
            except ValueError:
                df_input[col] = 0

    df_input[numeric_feature_cols] = scaler.transform(df_input[numeric_feature_cols])
    df_input = df_input[feature_columns]

    return float(model.predict(df_input)[0])
