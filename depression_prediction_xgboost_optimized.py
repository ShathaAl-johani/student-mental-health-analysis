# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load the cleaned dataset
df = pd.read_csv("cleaned_student_mental_health.csv")
df = df.drop(columns=["Timestamp"])  # Remove timestamp

# Encode categorical features
label_encoders = {}
categorical_columns = ["Choose your gender", "What is your course?", "Your current year of Study", "Marital status"]
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define features and target
X = df.drop(columns=["Do you have Depression?"])
y = df["Do you have Depression?"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[["Age", "What is your CGPA?"]] = scaler.fit_transform(X_train[["Age", "What is your CGPA?"]])
X_test[["Age", "What is your CGPA?"]] = scaler.transform(X_test[["Age", "What is your CGPA?"]])

# 🔹 Step 1: Define XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

# 🔹 Step 2: Define optimized parameter grid
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "gamma": [0, 0.1, 0.2, 0.3],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

# 🔹 Step 3: Optimize XGBoost using RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # زيادة عدد المحاولات للحصول على أفضل دقة
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# 🔹 Step 4: Get the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# 🔹 Step 5: Make Predictions and Evaluate the Optimized Model
y_pred = best_model.predict(X_test)

# Print Best Parameters
print(f"\n✅ Best Parameters: {random_search.best_params_}")

# Print Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Optimized XGBoost Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Optimized Confusion Matrix for Depression Prediction (XGBoost)")
plt.show()
