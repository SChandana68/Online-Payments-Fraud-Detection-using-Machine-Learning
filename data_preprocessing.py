import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("fraud_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nNull Values:")
print(df.isnull().sum())

print("\nFraud Distribution:")
print(df["Fraud"].value_counts())

# Plot fraud distribution
plt.figure()
sns.countplot(x="Fraud", data=df)
plt.title("Fraud vs Legitimate Transactions")
plt.show()

# -------------------------
# Encode categorical columns
# -------------------------
df_encoded = pd.get_dummies(df, columns=[
    "Payment_Method",
    "Merchant_Category",
    "Device_Type"
], drop_first=True)

# Save encoded dataset
df_encoded.to_csv("fraud_dataset_encoded.csv", index=False)

print("\nEncoded dataset saved successfully!")
