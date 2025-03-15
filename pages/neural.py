import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import matplotlib.pyplot as plt
from TrainModel.mlp import load_neural_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

st.title("📊 Neural Network Model (Load)")
st.subheader("🔍 ผลลัพธ์จากการ Train โมเดล")

# 🚀 ตรวจสอบว่ามีโมเดลหรือไม่ ถ้าไม่มีให้ Train ใหม่
if not os.path.exists("load_model.h5") or not os.path.exists("load_history.pkl"):
    st.warning("⏳ กำลัง Train ข้อมูล... โปรดรอสักครู่ 🚀")
    load_neural_model()  # Train อัตโนมัติ
    st.success("✅ โมเดล Train เสร็จแล้ว!")

# ✅ โหลดโมเดลที่ฝึกเสร็จแล้ว
model = keras.models.load_model("load_model.h5", custom_objects={"mse": keras.losses.MeanSquaredError()})

# ✅ โหลดประวัติการ Train
with open("load_history.pkl", "rb") as f:
    history = pickle.load(f)

# 🚀 1️⃣ แสดงโครงสร้างของโมเดลด้วย model.summary()
st.markdown("### 🔧 โครงสร้างของโมเดล")
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
model_summary = "\n".join(stringlist)
st.text(model_summary)

# Convert model summary to table format
summary_lines = model_summary.split('\n')
summary_lines = [line for line in summary_lines if line.strip() != '']
summary_data = []
for line in summary_lines[1:-4]:
    summary_data.append(line.split())

# Ensure all rows have the same number of columns
max_cols = max(len(row) for row in summary_data)
for row in summary_data:
    while len(row) < max_cols:
        row.append('')

summary_table = pd.DataFrame(summary_data, columns=summary_lines[0].split())
st.table(summary_table)

# 🚀 2️⃣ ทำนายผลจากชุด Test Set
st.markdown("### 🎯 ตัวอย่างการทำนายจาก Test Set")
df_test = pd.read_csv(r'Data_set/education_career_bad_model.csv')

# ✅ เตรียมข้อมูล (Data Cleaning) เช่นเดียวกับตอน Train
df_test_cleaned = df_test.drop(columns=["Student_ID", "Unnamed: 11"], errors="ignore")
for col in df_test_cleaned.select_dtypes(include=["float64", "int64"]).columns:
    df_test_cleaned[col].fillna(df_test_cleaned[col].mean(), inplace=True)
for col in df_test_cleaned.select_dtypes(include=["object"]).columns:
    df_test_cleaned[col].fillna(df_test_cleaned[col].mode()[0], inplace=True)

# แปลงข้อมูล Categorical โดยใช้ One-Hot Encoding
categorical_cols_test = df_test_cleaned.select_dtypes(include=['object']).columns.tolist()
df_test_cleaned = pd.get_dummies(df_test_cleaned, columns=categorical_cols_test, drop_first=True)

# โหลด scaler ที่บันทึกไว้
scaler_path = "TrainModel/scaler.pkl"
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None
    st.error("ไม่พบ scaler.pkl")

# เตรียมชุด Features สำหรับ Test Set
X_test = df_test_cleaned.drop(columns=["Starting_Salary"], errors="ignore")
if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
    y_test = np.log1p(df_test_cleaned["Starting_Salary"])

    # ทำนายผล
    y_pred = model.predict(X_test_scaled)

    # แสดงค่าจริงและค่าที่ทำนาย (เฉพาะ 10 แถวแรก)
    df_results = pd.DataFrame({
        "ค่าจริง (Actual Salary)": y_test.values[:10],
        "ค่าที่ทำนาย (Predicted Salary)": np.round(y_pred[:10].flatten(), 2)
    })
    st.dataframe(df_results)

    # # 🚀 3️⃣ วาดกราฟ Loss & MAE จาก Training
    # st.markdown("### 📈 กราฟ Training Loss & MAE")
    # fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # # กราฟ Loss
    # ax[0].plot(history["loss"], label="Train Loss")
    # ax[0].plot(history["val_loss"], label="Validation Loss")
    # ax[0].set_title("Training & Validation Loss")
    # ax[0].set_xlabel("Epochs")
    # ax[0].set_ylabel("Loss")
    # ax[0].legend()

    # # กราฟ MAE
    # ax[1].plot(history["mae"], label="Train MAE")
    # ax[1].plot(history["val_mae"], label="Validation MAE")
    # ax[1].set_title("Training & Validation MAE")
    # ax[1].set_xlabel("Epochs")
    # ax[1].set_ylabel("Mean Absolute Error")
    # ax[1].legend()

    # st.pyplot(fig)

    st.success("✅ โมเดล Train เสร็จเรียบร้อย! 🚀")

    # 🚀 คำนวณและแสดงผลการประเมินโมเดล
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    st.markdown("### 📊 Evaluating Model Performance")
    st.write(f"🔹 **Test Loss (MSE):** {test_loss:.4f}")
    st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"🔹 **Mean Absolute Percentage Error (MAPE):** {mape:.4f}")
    st.write(f"🔹 **R² Score (R2):** {r2:.4f}")

    df_metrics = pd.DataFrame({
        "Metric": ["Test Loss (MSE)", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)",
                   "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)", "R² Score (R2)"],
        "Value": [test_loss, mae, mse, rmse, mape, r2]
    })
    st.dataframe(df_metrics)
else:
    st.error("ไม่สามารถทำนายผลได้เนื่องจากไม่พบ scaler")