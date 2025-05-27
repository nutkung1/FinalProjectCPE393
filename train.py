import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


df = pd.read_csv('data/train80.csv')
columns_to_drop = [
    "Legislative District", 
    "Vehicle Location", 
    "Postal Code", 
    "City", 
    "2020 Census Tract", 
    "County", 
    "Electric Utility"
]

# Remove irrelevant columns
df = df.drop(columns=columns_to_drop)

# Filter out records where Electric Range is 0
df = df[df["Electric Range"] > 0]

# Define features and target
X = df.drop(columns=["Electric Range"]).fillna(df.mean(numeric_only=True))
y = df["Electric Range"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_reg = XGBRegressor(
    random_state=0, 
    booster='gbtree', 
    objective='reg:squarederror', 
    tree_method = "hist", 
    device = "cuda",
    colsample_bytree=1.0,
    learning_rate=0.660977898839114,
    max_depth=7,
    n_estimators=5000,
    reg_alpha=1e-09,
    reg_lambda=1e-09,
    subsample=1.0
)

best_reg.fit(X_train, y_train)
predictions = best_reg.predict(X_test)

print("Mean Squared Error : " + str(mean_squared_error(predictions, y_test)))
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))


rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse:.4f}")

mask = y_test != 0
if mask.any():
    mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
else:
    print("Mean Absolute Percentage Error: Cannot calculate (division by zero)")

with open("./Results/metrics.txt", "w") as outfile:
    outfile.write(f"Mean Squared Error: {mean_squared_error(predictions, y_test):.4f}\n")
    outfile.write(f"Mean Absolute Error: {mean_absolute_error(predictions, y_test):.4f}\n")
    outfile.write(f"Root Mean Squared Error: {rmse:.4f}\n")
    outfile.write(f"Mean Absolute Percentage Error: {mape:.2f}%")