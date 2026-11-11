import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =====================================
# Load Dataset
# =====================================

df = pd.read_excel("Dataset/manufacturing_data.xlsx")

# Features
X = df[['Temperature', 'Humidity', 'Vibration']]

# Target
y = df['Quality_Risk']

# Convert labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# =====================================
# XGBoost Model
# =====================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nXGBoost Accuracy: {accuracy:.4f}\n")

# =====================================
# Classification Report
# =====================================

report = classification_report(y_test, y_pred)

print(report)

# Save Accuracy
with open("Results/xgboost_accuracy.txt", "w") as f:
    f.write(f"XGBoost Accuracy: {accuracy:.4f}")

# Save Classification Report
with open("Results/xgboost_classification_report.txt", "w") as f:
    f.write(report)

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "Results/xgboost_confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# =====================================
# Save Model
# =====================================

joblib.dump(
    model,
    "Models/xgboost_model.pkl"
)

print("XGBoost Model Saved Successfully!")
print("Results Saved Successfully!")