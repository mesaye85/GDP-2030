import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Retrieve data from API
url = "https://api.worldbank.org/v2/countries/all/indicators/NY.GDP.MKTP.CD?format=json&date=2010:2021"
response = requests.get(url)
data = response.json()

# Store data in a pandas DataFrame
df = pd.DataFrame(data[1])
df.dropna(inplace=True)

# Train the model
X = df.drop("value", axis=1)
y = df["value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
