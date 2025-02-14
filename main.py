import requests
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Enable mixed precision for M1
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except:
    print("Mixed precision not supported on this device")

# Configure memory growth for M1
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    print("Memory growth setting not supported on this device")

VALID_COUNTRY_CODES = {
    "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "ITA",
    "JPN", "KOR", "MEX", "RUS", "ESP", "GBR", "USA"
}


def fetch_gdp_data():
    """Fetch GDP data with improved error handling."""
    indicators = {
        "GDP": "NY.GDP.MKTP.CD",
        "Inflation": "FP.CPI.TOTL.ZG",
        "Unemployment": "SL.UEM.TOTL.ZS",
        "Trade_GDP": "NE.TRD.GNFS.ZS",
        "FDI_GDP": "BX.KLT.DINV.WD.GD.ZS",
        "Interest_Rate": "FR.INR.RINR",
        "Gov_Debt": "GC.DOD.TOTL.GD.ZS",
        "Population": "SP.POP.TOTL"
    }

    data_frames = []
    for key, indicator in indicators.items():
        try:
            url = f"https://api.worldbank.org/v2/countries/all/indicators/{indicator}?format=json&date=1960:2023&per_page=20000"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = [
                {"year": int(entry["date"]),
                 "country_code": entry["countryiso3code"],
                 key: float(entry["value"])}
                for entry in data[1]
                if entry.get("countryiso3code") in VALID_COUNTRY_CODES
                   and entry.get("value") is not None
            ]

            df = pd.DataFrame(records)
            data_frames.append(df)

        except Exception as e:
            print(f"Error fetching {key}: {e}")
            continue

    final_df = data_frames[0]
    for df in data_frames[1:]:
        final_df = final_df.merge(df, on=["year", "country_code"], how="outer")

    return final_df


def process_data(df):
    """Process data with memory-efficient operations."""
    print("Processing data...")

    # Use more efficient operations
    df = df.copy()

    # Calculate basic metrics
    df['prev_gdp'] = df.groupby('country_code')['GDP'].shift(1)
    df['growth_rate'] = df.groupby('country_code')['GDP'].pct_change(fill_method=None)

    df['gdp_per_capita'] = df['GDP'] / df['Population']

    # Calculate rolling metrics efficiently
    for window in [3, 5, 10]:
        df[f'growth_rate_{window}y'] = df.groupby('country_code')['GDP'].pct_change(periods=window)
        df[f'rolling_growth_{window}y'] = df.groupby('country_code')['growth_rate'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # Log transformations
    df['log_gdp'] = np.log1p(df['GDP'])
    df['log_growth'] = np.log1p(df['growth_rate'])

    return df.dropna()


def prepare_sequences(df, seq_length=5):
    """Prepare sequences for LSTM with error handling for missing columns."""
    feature_cols = ['year', 'growth_rate_3y', 'growth_rate_5y', 'growth_rate_10y',
                    'rolling_growth_3y', 'rolling_growth_5y', 'rolling_growth_10y',
                    'Inflation', 'Unemployment', 'Trade_GDP', 'FDI_GDP', 'Interest_Rate',
                    'Gov_Debt', 'gdp_per_capita']

    #  Check if all required columns exist in df
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    #  Drop rows with NaNs before scaling
    df = df.dropna(subset=feature_cols)

    #  Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    #  Create sequences
    X, y = [], []
    for country in df['country_code'].unique():
        country_data = df[df['country_code'] == country]  # Filter by country
        scaled_country_data = scaled_features[df['country_code'] == country]

        for i in range(len(scaled_country_data) - seq_length):
            X.append(scaled_country_data[i:(i + seq_length)])
            y.append(country_data.iloc[i + seq_length]['log_gdp'])

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    """Build LSTM model optimized for M1."""
    model = Sequential([
        LSTM(32, activation='tanh', return_sequences=True,
             input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, activation='tanh'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    # Use Adam optimizer with adjusted learning rate for M1
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')

    return model


def train_model(X, y, seq_length=5):
    """Train model with M1 optimizations."""
    # Prepare sequences
    X_seq, y_seq, scaler = prepare_sequences(df, seq_length)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # Build model
    model = build_lstm_model((seq_length, X_train.shape[2]))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # Train with smaller batch size for M1
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model Performance - R²: {r2:.4f}, RMSE: {rmse:.4f}")

    return model, scaler, history


if __name__ == "__main__":
    print("Fetching GDP data...")
    df = fetch_gdp_data()

    print("Processing data...")
    df = process_data(df)

    print("Preparing sequences for LSTM...")
    X, y, scaler = prepare_sequences(df)  # Extract features and target

    print("Training LSTM model...")
    model, history = train_model(X, y)  # Ensure both `X` and `y` are passed

    # Save the trained model
    model.save("lstm_gdp_model_m1.keras")  # New Keras format
    joblib.dump(scaler, "scaler_m1.pkl")

    print("Model and artifacts saved successfully!")
