import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the Random Forest model pipeline
loaded_lr_model = joblib.load('linear_regression_model.joblib')

df = pd.read_csv("laptop_prices.csv")


X = df.drop('Price_euros', axis=1)
y = df['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions with the loaded model
predictions = loaded_lr_model.predict(X_test)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

print("Linear Regression Performance:")
evaluate_model(loaded_lr_model, X_test, y_test)

# Evaluate the loaded model to ensure it works as expected
