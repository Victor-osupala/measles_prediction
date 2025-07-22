import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create output directory
output_dir = "measles_model_output"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("synthetic_measles_dataset.csv")

# Features and target
X = df.drop("measles", axis=1)
y = df["measles"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, os.path.join(output_dir, "measles_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Save metrics in a text file
with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

# Save confusion matrix as image
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(output_dir, "confusion_matrix.jpeg"))

# Exploratory Data Analysis charts
# Fever distribution
plt.figure()
sns.histplot(df["fever"], bins=20, kde=True)
plt.title("Fever Distribution")
plt.savefig(os.path.join(output_dir, "fever_distribution.jpeg"))

# Age vs Measles
plt.figure()
sns.boxplot(x="measles", y="age", data=df)
plt.title("Age Distribution by Measles Status")
plt.savefig(os.path.join(output_dir, "age_vs_measles.jpeg"))

# Immunization vs Measles
plt.figure()
sns.countplot(x="immunization_status", hue="measles", data=df)
plt.title("Immunization Status vs Measles")
plt.savefig(os.path.join(output_dir, "immunization_vs_measles.jpeg"))

print("Training complete. Artifacts saved in:", output_dir)
