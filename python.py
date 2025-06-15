import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_wb_forecast_data(start_year=2010, end_year=2030):
    """Fetch World Bank forecast data and historical data."""
    indicators = {
        'GDP': 'NY.GDP.MKTP.CD',  # GDP (current US$)
        'GDP_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
        'GDP_pcap': 'NY.GDP.PCAP.CD',  # GDP per capita (current US$)
        'GDP_pcap_growth': 'NY.GDP.PCAP.KD.ZG',  # GDP per capita growth (annual %)
        'GNI': 'NY.GNP.MKTP.CD',  # GNI (current US$)
        'GNI_pcap': 'NY.GNP.PCAP.CD',  # GNI per capita (current US$)
    }

    all_data = []

    try:
        for indicator_name, indicator_code in indicators.items():
            # World Bank API v2 endpoint
            url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
            params = {
                'format': 'json',
                'date': f"{start_year}:{end_year}",
                'per_page': 20000
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data or len(data) < 2:
                logger.warning(f"No data received for {indicator_name}")
                continue

            # Extract data and create DataFrame
            records = []
            for entry in data[1]:
                if entry['value'] is not None:
                    record = {
                        'country': entry['country']['value'],
                        'country_code': entry['country']['id'],
                        'year': int(entry['date']),
                        'indicator': indicator_name,
                        'value': float(entry['value']),
                        'is_forecast': int(entry['date']) > 2023  # Mark as forecast if after 2023
                    }
                    records.append(record)

            if records:  # Only create DataFrame if we have records
                df = pd.DataFrame(records)
                all_data.append(df)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching World Bank data: {e}")
        raise

    if not all_data:  # Check if we have any DataFrames in our list
        raise ValueError("No data was successfully fetched from any indicator")

    # Combine all DataFrames
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

def fetch_imf_forecast_data():
    """Fetch IMF WEO forecast data."""
    try:
        # IMF WEO data endpoint (using latest available)
        url = "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH"  # Real GDP growth rate
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Process IMF forecast data
        records = []

        # Get the available years from the values
        if 'dates' in data:
            years = [year for year in data['dates'] if year.isdigit()]
        else:
            logger.warning("No dates found in IMF data")
            return pd.DataFrame()

        for country_code, values in data.get('values', {}).items():
            # Skip if country_code is not a string (sometimes metadata fields appear)
            if not isinstance(country_code, str):
                continue

            for year in years:
                if year in values and values[year] is not None:
                    try:
                        records.append({
                            'country_code': country_code,
                            'year': int(year),
                            'value': float(values[year]),
                            'indicator': 'GDP_growth_imf',
                            'is_forecast': int(year) > 2023
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse value for country {country_code}, year {year}: {e}")
                        continue

        if not records:
            logger.warning("No valid IMF forecast data available")
            return pd.DataFrame()

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching IMF forecast data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error processing IMF data: {e}")
        return pd.DataFrame()

def prepare_forecast_features(df, imf_data=None):
    """Prepare features with focus on forecast reliability."""
    logger.info("Preparing forecast features...")

    if df.empty:
        raise ValueError("Empty DataFrame provided for feature preparation")

    # Pivot the data to have indicators as columns
    pivot_df = df.pivot_table(
        index=['country', 'country_code', 'year', 'is_forecast'],
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Sort by country and year
    pivot_df = pivot_df.sort_values(['country_code', 'year'])

    # Create separate DataFrames for historical and forecast data
    historical_data = pivot_df[~pivot_df['is_forecast']].copy()
    forecast_data = pivot_df[pivot_df['is_forecast']].copy()

    if historical_data.empty:
        raise ValueError("No historical data available for feature preparation")

    # Calculate historical trends and patterns
    for country in pivot_df['country_code'].unique():
        country_mask = historical_data['country_code'] == country
        country_data = historical_data[country_mask]

        if len(country_data) > 0 and 'GDP_growth' in country_data.columns:
            # Calculate historical growth patterns
            historical_data.loc[country_mask, 'hist_growth_mean'] = country_data['GDP_growth'].expanding().mean()
            historical_data.loc[country_mask, 'hist_growth_std'] = country_data['GDP_growth'].expanding().std()

            # Calculate 5-year moving averages
            historical_data.loc[country_mask, 'growth_ma_5'] = country_data['GDP_growth'].rolling(window=5, min_periods=1).mean()

            # Calculate growth volatility
            historical_data.loc[country_mask, 'growth_volatility'] = country_data['GDP_growth'].rolling(window=5, min_periods=1).std()

    # Merge IMF forecasts if available
    if not imf_data.empty:
        # Pivot IMF data if needed
        if 'indicator' in imf_data.columns:
            imf_pivot = imf_data.pivot_table(
                index=['country_code', 'year', 'is_forecast'],
                columns='indicator',
                values='value',
                aggfunc='first'
            ).reset_index()

            # Merge with forecast data
            forecast_data = forecast_data.merge(
                imf_pivot,
                on=['country_code', 'year', 'is_forecast'],
                how='left',
                suffixes=('', '_imf')
            )
        else:
            # If IMF data is already pivoted
            forecast_data = forecast_data.merge(
                imf_data,
                on=['country_code', 'year', 'is_forecast'],
                how='left',
                suffixes=('', '_imf')
            )

    # Combine historical and forecast data
    final_df = pd.concat([historical_data, forecast_data], ignore_index=True)
    final_df = final_df.sort_values(['country_code', 'year'])

    return final_df

def train_forecast_model(df):
    """Train model with emphasis on forecast accuracy."""
    if df.empty:
        raise ValueError("Empty DataFrame provided for model training")

    # Prepare features
    feature_cols = [col for col in df.columns if col not in 
                   ['country', 'country_code', 'year', 'GDP', 'is_forecast']]

    # Split data into historical and forecast periods
    historical_data = df[~df['is_forecast']]
    forecast_data = df[df['is_forecast']]

    if historical_data.empty:
        raise ValueError("No historical data available for model training")

    # Handle NaN values
    logger.info("Handling missing values in the data...")

    # Check if GDP column exists and has non-NaN values
    if 'GDP' not in historical_data.columns or historical_data['GDP'].isna().all():
        raise ValueError("GDP column is missing or contains only NaN values")

    # Drop rows where GDP is NaN
    historical_data = historical_data.dropna(subset=['GDP'])

    # For feature columns, fill NaN values with the median of each column
    for col in feature_cols:
        if col in historical_data.columns:
            if historical_data[col].isna().any():
                median_value = historical_data[col].median()
                # If median is NaN (all values are NaN), use 0
                if pd.isna(median_value):
                    median_value = 0
                historical_data[col] = historical_data[col].fillna(median_value)

    # Select only columns that exist in the DataFrame
    valid_feature_cols = [col for col in feature_cols if col in historical_data.columns]

    # Log the number of features being used
    logger.info(f"Using {len(valid_feature_cols)} features for training")

    X = historical_data[valid_feature_cols]
    y = historical_data['GDP']

    # Use TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)

    models = []
    metrics = []

    for train_idx, test_idx in tscv.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model with more trees and deeper analysis
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics.append({
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        })

        models.append((model, scaler))

    # Log metrics
    avg_metrics = pd.DataFrame(metrics).mean()
    logger.info(f"Average metrics across folds: {avg_metrics}")

    # Return the best model based on RMSE
    best_idx = np.argmin([m['rmse'] for m in metrics])
    return models[best_idx]

if __name__ == "__main__":
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Fetch both World Bank and IMF data
        logger.info("Fetching World Bank forecast data...")
        wb_data = fetch_wb_forecast_data()

        logger.info("Fetching IMF forecast data...")
        imf_data = fetch_imf_forecast_data()

        # Combine and prepare data
        logger.info("Preparing combined forecast data...")
        all_data = prepare_forecast_features(wb_data, imf_data)

        # Train model
        logger.info("Training forecast model...")
        model, scaler = train_forecast_model(all_data)

        # Save artifacts
        logger.info("Saving model and artifacts...")
        import joblib
        joblib.dump(model, 'models/gdp_forecast_model.pkl')
        joblib.dump(scaler, 'models/gdp_forecast_scaler.pkl')

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
