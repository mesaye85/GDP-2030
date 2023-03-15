import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Retrieve data from API
url = "https://api.worldbank.org/v2/countries/all/indicators/NY.GDP.MKTP.CD?format=json&date=2010:2021&per_page=20000"
response = requests.get(url)
data = response.json()

# Store data in a pandas DataFrame
df = pd.DataFrame(data[1])
df.dropna(inplace=True)

# Extract year and country code
df['year'] = df['date'].astype(int)
df['country_code'] = df['countryiso3code']

# Encode country codes as numerical values
le = LabelEncoder()
df['country_code'] = le.fit_transform(df['country_code'])

# Prepare data for modeling
X = df[['year', 'country_code']]
y = df["value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
