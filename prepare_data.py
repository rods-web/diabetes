import pandas as pd

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("data/diabetes_raw.csv", header=None, names=columns)
df.to_csv("data/diabetes.csv", index=False)
print(f"Dataset ready: {df.shape}")
print(df.head())
