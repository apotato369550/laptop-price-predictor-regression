import pandas as pd
import numpy as np
import joblib

# For data visualization (optional)
# import matplotlib.pyplot as plt
# import seaborn as sns

# For preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression models
from sklearn.linear_model import LinearRegression

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("laptop_prices.csv")

# Display initial data information
print(df.head())
print(df.info())
print(df.describe())

# Check the shape of the data
print(f"Data shape: {df.shape}")

# Check for missing values
print(df.isnull().sum())

# Get summary statistics for numerical features
print(df.describe())

# (Optional) Visualize the distribution of the target variable
'''
sns.histplot(df['Price_euros'], kde=True)
plt.title('Distribution of Price in Euros')
plt.show()

# Correlation matrix (for numerical features)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
'''

# Define categorical and numerical columns
categorical_cols = [
    'Company', 'Product', 'TypeName', 'OS', 'Touchscreen',
    'IPSpanel', 'RetinaDisplay', 'CPU_company', 'CPU_model',
    'PrimaryStorageType', 'SecondaryStorageType', 'GPU_company', 'GPU_model', 'Screen'
]

numerical_cols = [
    'Inches', 'Ram', 'Weight', 'ScreenW',
    'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage'
]

# Features and target
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']

# Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline with preprocessing and the model
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline_lr.fit(X_train, y_train)

# Save the trained Linear Regression model pipeline
joblib.dump(pipeline_lr, 'linear_regression_model.joblib')

print("Model training and saving completed successfully.")
