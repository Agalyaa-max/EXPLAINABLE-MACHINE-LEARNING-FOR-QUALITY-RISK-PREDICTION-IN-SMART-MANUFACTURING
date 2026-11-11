import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Dataset
df = pd.read_excel("Dataset/manufacturing_data.xlsx")

# Features and Target
X = df[['Temperature', 'Humidity', 'Vibration']]
y = df['Quality_Risk']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save Model
joblib.dump(model, "Models/random_forest_model.pkl")

# Classification Report
report = classification_report(y_test, y_pred)

with open("Results/classification_report.txt", "w") as f:
    f.write(report)

# Accuracy File
with open("Results/accuracy_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Results/confusion_matrix.png")
plt.close()

print("Model Saved Successfully!")
print("Results Generated Successfully!")