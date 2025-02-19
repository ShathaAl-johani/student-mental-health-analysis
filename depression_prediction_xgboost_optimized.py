# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # معالجة عدم التوازن في البيانات
import xgboost as xgb  # مكتبة XGBoost

# Load the cleaned dataset
df = pd.read_csv("cleaned_student_mental_health.csv")

# Remove the timestamp column (not needed for prediction)
df = df.drop(columns=["Timestamp"])

# Convert categorical variables into numeric values using Label Encoding
label_encoders = {}
categorical_columns = ["Choose your gender", "What is your course?", "Your current year of Study", "Marital status"]

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define features (X) and target variable (y)
X = df.drop(columns=["Do you have Depression?"])  # Features
y = df["Do you have Depression?"]  # Target variable (Depression)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize numerical features (CGPA, Age)
scaler = StandardScaler()
X_train[["Age", "What is your CGPA?"]] = scaler.fit_transform(X_train[["Age", "What is your CGPA?"]])
X_test[["Age", "What is your CGPA?"]] = scaler.transform(X_test[["Age", "What is your CGPA?"]])

# 🔹 Step 2: Optimize XGBoost using GridSearchCV

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "gamma": [0, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# 🔹 Step 3: Make Predictions and Evaluate the Optimized Model
y_pred = best_model.predict(X_test)

# Print Best Parameters
print(f"\n Best Parameters: {grid_search.best_params_}")

# Print Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Optimized XGBoost Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Optimized Confusion Matrix for Depression Prediction (XGBoost)")
plt.show()

