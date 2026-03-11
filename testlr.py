import joblib
import pandas as pd
import numpy as np

# 1. Load model AND feature names
model = joblib.load('churn_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # ✅ Critical!

print(f"Model expects these {len(feature_names)} features: {feature_names}")

# 2. Create new customer data (ALL features you have)
new_customer_2 = {
    'tenure': 60,
    'age': 55,
    'address': 20,
    'income': 150,
    'ed': 4,
    'employ': 25,
    'equip': 1,
    'callcard': 1,
    'wireless': 1,
    'longmon': 25.0,
    'tollmon': 40.0,
    'equipmon': 30.0,
    'cardmon': 20.0,
    'wiremon': 50.0,
    'longten': 1500,
    'tollten': 2000,
    'cardten': 1000,
    'voice': 1,
    'pager': 1,
    'internet': 1,
    'callwait': 1,
    'confer': 1,
    'ebill': 1,
    'loglong': 3.2,
    'logtoll': 3.6,
    'lninc': 5.0,
    'custcat': 4
}

# 3. Convert to DataFrame
new_df = pd.DataFrame([new_customer_2])

# 4. ✅ Select ONLY the features the model was trained on (in correct order!)
X_new = new_df[feature_names]

print(f"\n✅ Using {len(X_new.columns)} features: {X_new.columns.tolist()}")
print(f"Input data:\n{X_new}")

# 5. Get predictions
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[:, 1]

# 6. ✅ Apply optimized threshold (0.3 instead of 0.5)
threshold = 0.3
predicted_churn = (probability >= threshold).astype(int)

# 7. Display results with risk level
print(f"\n" + "="*50)
print("🎯 CHURN PREDICTION RESULTS")
print("="*50)
print(f"Churn Probability: {probability[0]:.2%}")
print(f"Threshold Used:    {threshold}")
print(f"Predicted Churn:   {'YES ⚠️' if predicted_churn[0] == 1 else 'NO ✅'}")

# Add risk level
if probability[0] >= 0.7:
    risk = "HIGH 🔴"
    action = "URGENT: Assign to retention specialist"
elif probability[0] >= 0.5:
    risk = "MEDIUM 🟡"
    action = "Offer personalized discount"
elif probability[0] >= threshold:
    risk = "LOW 🟢"
    action = "Send engagement email"
else:
    risk = "NONE ⚪"
    action = "No action needed"

print(f"Risk Level:        {risk}")
print(f"Recommended Action: {action}")
print("="*50)