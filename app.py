import sqlite3
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# IMPORTANT: Same columns used during training
training_columns = joblib.load("training_columns.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    try:
        # ----------- Get Form Data -----------
        user_id = request.form["User_ID"]
        amount = float(request.form["Amount"])
        payment = request.form["Payment_Method"]
        merchant = request.form["Merchant_Category"]
        international = int(request.form["Is_International"])
        device = request.form["Device_Type"]
        account_age = int(request.form["Account_Age_Months"])
        previous_fraud = int(request.form["Previous_Fraud"])

        # ----------- Create DataFrame -----------
        input_dict = {
            "Amount": amount,
            "Is_International": international,
            "Account_Age_Months": account_age,
            "Previous_Fraud": previous_fraud,
            "Payment_Method": payment,
            "Merchant_Category": merchant,
            "Device_Type": device
        }

        df = pd.DataFrame([input_dict])

        # One-hot encoding
        df = pd.get_dummies(df)

        # Ensure same columns as training
        df = df.reindex(columns=training_columns, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # ----------- Predict -----------
        prob = model.predict_proba(df_scaled)[0][1]
        risk_percent = round(prob * 100, 2)

        # ----------- Risk Level -----------
        if risk_percent < 40:
            risk_level = "LOW"
            decision = "APPROVED"
            account_status = "ACTIVE"

        elif risk_percent < 70:
            risk_level = "MEDIUM"
            decision = "UNDER REVIEW"
            account_status = "ACTIVE"

        else:
            risk_level = "HIGH"
            decision = "BLOCKED"
            account_status = "SUSPENDED"

        # ----------- Connect to Database -----------
        conn = sqlite3.connect("fraud_detection.db")
        cursor = conn.cursor()

        # ----------- Insert into Database -----------
        cursor.execute("""
        INSERT INTO transactions (
            user_id, amount, payment, merchant, international,
            device, account_age, previous_fraud,
            risk_percent, risk_level,
            decision, account_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, amount, payment, merchant, international,
            device, account_age, previous_fraud,
            risk_percent, risk_level,
            decision, account_status
        ))

        conn.commit()
        conn.close()

        return render_template("index.html",
                               risk=risk_percent,
                               level=risk_level,
                               decision=decision)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
