import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("data/diabetes.csv")
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
for col in zero_as_missing:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = joblib.load("models/scaler.pkl")
feature_mask = joblib.load("models/feature_mask.pkl")
best_rf = joblib.load("models/enhanced_rf.pkl")

X_train_s = scaler.transform(X_train)[:, feature_mask]
X_test_s = scaler.transform(X_test)[:, feature_mask]
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

plt.figure(figsize=(7, 6))
y_proba = best_rf.predict_proba(X_test_s)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color="#667eea", lw=2.5, label=f"Enhanced RF (AUC = {auc(fpr, tpr):.3f})")
for name, m in {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "SVM": SVC(kernel="rbf", probability=True, random_state=42)}.items():
    m.fit(X_train_bal, y_train_bal)
    f, t, _ = roc_curve(y_test, m.predict_proba(X_test_s)[:, 1])
    plt.plot(f, t, lw=1.8, label=f"{name} (AUC = {auc(f, t):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison", fontweight="bold")
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.savefig("results/auc_roc_curve.png"); plt.close()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_rf.predict(X_test_s))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            cbar=False, annot_kws={"size": 16, "weight": "bold"})
plt.title("Confusion Matrix - Enhanced RF", fontweight="bold")
plt.savefig("results/confusion_matrix.png"); plt.close()

imp = pd.read_csv("models/feature_importance.csv").sort_values("Importance")
plt.figure(figsize=(8, 5))
plt.barh(imp["Feature"], imp["Importance"], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp))))
plt.xlabel("Importance Score")
plt.title("Feature Importance (Gini) - Random Forest", fontweight="bold")
plt.savefig("results/feature_importance.png"); plt.close()

rdf = pd.read_csv("models/comparison_results.csv")
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(rdf)); w = 0.15
plt.figure(figsize=(11, 5.5))
for i, metric in enumerate(metrics):
    plt.bar(x + i * w, rdf[metric], w, label=metric)
plt.xticks(x + w * 2, rdf["Model"], rotation=15, ha="right")
plt.ylim(0, 1.0); plt.legend(loc="lower right"); plt.grid(axis="y", alpha=0.3)
plt.title("Model Performance Comparison", fontweight="bold")
plt.savefig("results/model_comparison.png"); plt.close()

print("All figures saved to results/")
