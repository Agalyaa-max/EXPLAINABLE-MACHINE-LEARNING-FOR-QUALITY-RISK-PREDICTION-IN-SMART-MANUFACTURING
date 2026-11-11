import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_excel("Dataset/manufacturing_data.xlsx")

X = df[['Temperature', 'Humidity', 'Vibration']]

# Load Model
model = joblib.load("Models/random_forest_model.pkl")

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("SHAP/shap_summary.png", bbox_inches="tight")
plt.close()

print("SHAP Summary Plot Generated Successfully!")