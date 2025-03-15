import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from TrainModel.mlp import load_neural_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

st.title("📊 Neural Network Model (load)")
st.subheader("🔍 ผลลัพธ์จากการ Train โมเดล")

# 🚀 ตรวจสอบว่ามีโมเดลหรือไม่ ถ้าไม่มีให้ Train ใหม่
if not os.path.exists("load_model.h5") or not os.path.exists("load_history.pkl"):
    st.warning("⏳ กำลัง Train ข้อมูล... โปรดรอสักครู่ 🚀")
    load_neural_model()  # ✅ Train อัตโนมัติ
    st.success("✅ โมเดล Train เสร็จแล้ว!")

# ✅ โหลดโมเดล และกำหนด loss function ใหม่
model = keras.models.load_model("load_model.h5", custom_objects={"mse": keras.losses.MeanSquaredError()})

# ✅ โหลดประวัติการ Train
with open("load_history.pkl", "rb") as f:
    history = pickle.load(f)

# 🚀 1️⃣ แสดงโครงสร้างของโมเดล
st.markdown("### 🔧 โครงสร้างของโมเดล")
st.text("\n".join([f"{layer.name} - {layer.output.shape}" for layer in model.layers]))

# 🚀 2️⃣ ทำนายผลจากชุดทดสอบ
st.markdown("### 🎯 ตัวอย่างการทำนายจาก Test Set")
df = pd.read_csv(r'Data_set\education_career_realmodel.csv')

# ✅ เตรียมข้อมูลเหมือนตอน Train
df_cleaned = df.drop(columns=["Student_ID", "Unnamed: 11"], errors="ignore")

for col in df_cleaned.select_dtypes(include=["float64", "int64"]).columns:
    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

for col in df_cleaned.select_dtypes(include=["object"]).columns:
    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

from sklearn.preprocessing import StandardScaler, LabelEncoder
label_encoders = {}
for col in df_cleaned.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

X = df_cleaned.drop(columns=["Starting_Salary"])
y = np.log1p(df_cleaned["Starting_Salary"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

# ✅ แสดงค่าจริงและค่าทำนายบางส่วน
df_results = pd.DataFrame({
    "ค่าจริง (Actual Salary)": y_test.values[:10],
    "ค่าที่ทำนาย (Predicted Salary)": np.round(y_pred[:10].flatten(), 2)
})
st.dataframe(df_results)

# 🚀 3️⃣ วาดกราฟ Loss & MAE จาก Training
import matplotlib.pyplot as plt
st.markdown("### 📈 กราฟ Training Loss & MAE")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# ✅ กราฟ Loss
ax[0].plot(history["loss"], label="Train Loss")
ax[0].plot(history["val_loss"], label="Validation Loss")
ax[0].set_title("Training & Validation Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

# ✅ กราฟ MAE
ax[1].plot(history["mae"], label="Train MAE")
ax[1].plot(history["val_mae"], label="Validation MAE")
ax[1].set_title("Training & Validation MAE")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Mean Absolute Error")
ax[1].legend()

st.pyplot(fig)

st.success("✅ โมเดล Train เสร็จเรียบร้อย! 🚀")

# 🚀 คำนวณค่าประเมินผล
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)  # ใช้ evaluate() ของโมเดล
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 🚀 แสดงผลการประเมิน
st.markdown("### 📊 Evaluating Model Performance")
st.write(f"🔹 **Test Loss (MSE) :** {test_loss:.4f}")
st.write(f"🔹 **Mean Absolute Error (MAE) :** {mae:.4f}")
st.write(f"🔹 **Mean Squared Error (MSE) :** {mse:.4f}")
st.write(f"🔹 **Root Mean Squared Error (RMSE) :** {rmse:.4f}")
st.write(f"🔹 **Mean Absolute Percentage Error (MAPE) :** {mape:.4f}")
st.write(f"🔹 **R² Score (R2) :** {r2:.4f}")

# 🚀 แสดงผลแบบ DataFrame
df_metrics = pd.DataFrame({
    "Metric": ["Test Loss (MSE)", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)", "R² Score (R2)"],
    "Value": [test_loss, mae, mse, rmse, mape, r2]
})
st.dataframe(df_metrics)

