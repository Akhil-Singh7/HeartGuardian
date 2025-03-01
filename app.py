import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

st.title("HeartGuardian")

def user_input():
    age = st.number_input("Age", min_value=1, max_value=120, value=56)
    sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=236)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=178)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", value=0.8)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1=normal; 2=fixed; 3=reversable)", [1, 2, 3])

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]])
    
    return features

features = user_input()

if st.button("Predict"):
    scaled = scaler.transform(features)
    

    prediction = model.predict(scaled)
    st.write("Prediction Output (0 = No disease, 1 = Disease):", prediction)

    result = "🚨 Likely to have heart disease." if prediction[0] == 1 else "✅ Unlikely to have heart disease."
    st.subheader("Final Result:")
    st.success(result)
