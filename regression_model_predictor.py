import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Linear Regression model pipeline
loaded_lr_model = joblib.load('linear_regression_model.joblib')

# Load the dataset
df = pd.read_csv("laptop_prices.csv")

# Define features and target
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']

# Split the data into training and testing sets (consistent with training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the evaluation function
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

print("Loaded Linear Regression Performance:")
evaluate_model(loaded_lr_model, X_test, y_test)

# Function to input new laptop data
def get_new_laptop_data():
    print("\nEnter the specifications for the new laptop:")
    
    # Define the order and type of inputs based on your features
    new_data = {}
    
    # Categorical Features
    new_data['Company'] = input("Company (e.g., Apple, HP, Dell): ")
    new_data['Product'] = input("Product (e.g., MacBook Pro, HP 250 G6): ")
    new_data['TypeName'] = input("TypeName (e.g., Ultrabook, Notebook): ")
    new_data['OS'] = input("Operating System (e.g., macOS, No OS): ")
    new_data['Touchscreen'] = input("Touchscreen (Yes/No): ")
    new_data['IPSpanel'] = input("IPSpanel (Yes/No): ")
    new_data['RetinaDisplay'] = input("RetinaDisplay (Yes/No): ")
    new_data['CPU_company'] = input("CPU Company (e.g., Intel, AMD): ")
    new_data['CPU_model'] = input("CPU Model (e.g., Core i5, Core i7): ")
    new_data['PrimaryStorageType'] = input("Primary Storage Type (e.g., SSD, Flash Storage): ")
    new_data['SecondaryStorageType'] = input("Secondary Storage Type (e.g., No): ")
    new_data['GPU_company'] = input("GPU Company (e.g., Intel, AMD): ")
    new_data['GPU_model'] = input("GPU Model (e.g., Iris Plus Graphics 640): ")
    new_data['Screen'] = input("Screen Type (e.g., Full HD, Standard): ")
    
    # Numerical Features with input validation
    while True:
        try:
            new_data['Inches'] = float(input("Screen Size (Inches): "))
            break
        except ValueError:
            print("Please enter a valid number for Inches.")
    
    while True:
        try:
            new_data['Ram'] = int(input("RAM (GB): "))
            break
        except ValueError:
            print("Please enter a valid integer for RAM.")
    
    while True:
        try:
            new_data['Weight'] = float(input("Weight (kg): "))
            break
        except ValueError:
            print("Please enter a valid number for Weight.")
    
    while True:
        try:
            new_data['ScreenW'] = int(input("Screen Width (pixels): "))
            break
        except ValueError:
            print("Please enter a valid integer for Screen Width.")
    
    while True:
        try:
            new_data['ScreenH'] = int(input("Screen Height (pixels): "))
            break
        except ValueError:
            print("Please enter a valid integer for Screen Height.")
    
    while True:
        try:
            new_data['CPU_freq'] = float(input("CPU Frequency (GHz): "))
            break
        except ValueError:
            print("Please enter a valid number for CPU Frequency.")
    
    while True:
        try:
            new_data['PrimaryStorage'] = int(input("Primary Storage (GB): "))
            break
        except ValueError:
            print("Please enter a valid integer for Primary Storage.")
    
    while True:
        try:
            new_data['SecondaryStorage'] = int(input("Secondary Storage (GB): "))
            break
        except ValueError:
            print("Please enter a valid integer for Secondary Storage.")
    
    # Convert to DataFrame
    new_laptop_df = pd.DataFrame([new_data])
    
    return new_laptop_df

# Function to make prediction on new data
def predict_new_laptop(model):
    new_laptop = get_new_laptop_data()
    predicted_price = model.predict(new_laptop)[0]
    print(f"\nPredicted Price for the entered laptop: €{predicted_price:.2f}")

# Function to compare actual vs predicted on random samples
def compare_random_samples(model, X_test, y_test, num_samples=5):
    # Randomly select indices
    random_indices = np.random.choice(X_test.index, size=num_samples, replace=False)
    
    # Extract the corresponding rows
    X_random = X_test.loc[random_indices]
    y_random = y_test.loc[random_indices]
    
    # Make predictions
    predictions_random = model.predict(X_random)
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual Price (€)': y_random,
        'Predicted Price (€)': predictions_random,
        'Difference (€)': predictions_random - y_random
    }).reset_index(drop=True)
    
    # Optionally, include some features
    features_random = X_random.reset_index(drop=True).copy()
    comparison_full_df = pd.concat([features_random, comparison_df], axis=1)
    
    print(f"\nComparison of Actual vs Predicted Prices for {num_samples} Random Samples:")
    print(comparison_full_df)
    
    # Visualize the comparison
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    indices = np.arange(num_samples)
    
    plt.bar(indices, comparison_full_df['Actual Price (€)'], bar_width, label='Actual', color='blue')
    plt.bar(indices + bar_width, comparison_full_df['Predicted Price (€)'], bar_width, label='Predicted', color='orange')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Price (€)')
    plt.title(f'Actual vs Predicted Prices for {num_samples} Random Samples')
    plt.xticks(indices + bar_width / 2, comparison_full_df.index + 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Execute comparison on random samples
compare_random_samples(loaded_lr_model, X_test, y_test, num_samples=5)

# Predict on a new laptop
predict_new_laptop(loaded_lr_model)
