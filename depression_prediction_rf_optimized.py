import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the cleaned dataset
df = pd.read_csv("cleaned_student_mental_health.csv")

#  Encode categorical variables
categorical_features = ["Choose your gender", "What is your course?", "Your current year of Study", "Marital status"]
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#  Features and target variable
X = df[["Age", "What is your CGPA?"] + categorical_features]
y = df["Do you have Depression?"]

#  Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Standardize numerical features
scaler = StandardScaler()
X_train[["Age", "What is your CGPA?"]] = scaler.fit_transform(X_train[["Age", "What is your CGPA?"]])
X_test[["Age", "What is your CGPA?"]] = scaler.transform(X_test[["Age", "What is your CGPA?"]])

#  Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

#  Hyperparameter grid for tuning
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

#  Perform Randomized Search
random_search = RandomizedSearchCV(
    rf_model, 
    param_distributions=param_grid,
    n_iter=20, 
    scoring="accuracy", 
    cv=5, 
    random_state=42, 
    n_jobs=-1
)

random_search.fit(X_train, y_train)

#  Get the best model
best_rf_model = random_search.best_estimator_

#  Make predictions
y_pred = best_rf_model.predict(X_test)

#  Print results
print("\n Best Parameters:", random_search.best_params_)
print("\n Optimized Random Forest Accuracy:", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

#  Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Optimized Confusion Matrix for Depression Prediction (Random Forest)")
plt.show()
