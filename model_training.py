import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load encoded dataset
df = pd.read_csv("fraud_dataset_encoded.csv")

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 🔥 Use class_weight
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_sm, y_train_sm)

# Predict
y_probs = model.predict_proba(X_test)[:, 1]

# 🔥 Adjust threshold here
threshold = 0.35
y_pred = (y_probs > threshold).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "fraud_model.pkl")

print("\nModel saved successfully!")
