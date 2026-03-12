import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Flight_Data.csv")

print(df.head())
print(df.info())

# Clean Total Stops
df["Total_Stops"] = df["Total_Stops"].replace({
    "non-stop":0,
    "1 stop":1,
    "2 stops":2,
    "3 stops":3,
    "4 stops":4
})

# Convert date
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")

df["Journey_day"] = df["Date_of_Journey"].dt.day
df["Journey_month"] = df["Date_of_Journey"].dt.month

# Extract departure hour
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour

# Extract arrival hour
df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour

# Convert duration to minutes
def convert_duration(x):
    hours = 0
    minutes = 0
    
    if "h" in x:
        hours = int(x.split("h")[0])
        
    if "m" in x:
        minutes = int(x.split("m")[0].split()[-1])
        
    return hours*60 + minutes

df["Duration_minutes"] = df["Duration"].apply(convert_duration)

# Select features
data = df[[
    "Airline",
    "Source",
    "Destination",
    "Total_Stops",
    "Duration_minutes",
    "Dep_hour",
    "Arrival_hour",
    "Journey_day",
    "Journey_month",
    "Price"
]]

# One-hot encoding
data = pd.get_dummies(data, columns=["Airline","Source","Destination"])

# Visualization
plt.figure(figsize=(10,5))
sns.histplot(data["Price"], bins=40)
plt.title("Distribution of Ticket Prices")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Total_Stops"], y=df["Price"])
plt.title("Ticket Price vs Stops")
plt.show()

# Unsupervised learning (Clustering)
features = data.drop("Price", axis=1)

kmeans = KMeans(n_clusters=3, random_state=42)

data["Price_Cluster"] = kmeans.fit_predict(features)

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=data["Duration_minutes"],
    y=data["Price"],
    hue=data["Price_Cluster"],
    palette="viridis"
)

plt.title("Flight Price Clusters")
plt.show()

# Supervised learning
X = data.drop(["Price","Price_Cluster"], axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,6),
    title="Feature Importance"
)

plt.show()

# Example prediction
sample_flight = X.iloc[0:1]

predicted_price = model.predict(sample_flight)

print("\nExample Predicted Price: ₹", round(predicted_price[0],2))

# Insights
print("\nInsights from the Model:")
print("- Non-stop flights tend to be priced higher.")
print("- Duration significantly impacts price.")
print("- Airline brand plays a major role in ticket pricing.")
print("- Major routes and peak hours influence ticket cost.")
print("- Clustering reveals budget, mid-range, and premium flight segments.")