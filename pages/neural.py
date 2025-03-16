import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os , re
import matplotlib.pyplot as plt
from TrainModel.mlp import load_neural_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

MODEL_DIR = "TrainModel"
MODEL_PATH = os.path.join(MODEL_DIR, "load_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "train_columns.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "load_history.pkl")

st.title("📊 Neural Network Model")
st.subheader("ผลลัพธ์จากการ Train โมเดล")

if not os.path.exists(MODEL_PATH) or not os.path.exists(HISTORY_PATH):
    st.warning("⏳ กำลัง Train ข้อมูล... โปรดรอสักครู่ 🚀")
    load_neural_model()  # Train อัตโนมัติ
    st.success("✅ โมเดล Train เสร็จแล้ว!")

#โหลดโมเดลที่ฝึกเสร็จแล้ว
model = keras.models.load_model(MODEL_PATH, custom_objects={"mse": keras.losses.MeanSquaredError()})
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

#โหลดประวัติการ Train
with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

#แสดงโครงสร้างของโมเดล
st.markdown("### 🔧 โครงสร้างของโมเดล")
stringlist = []  
model.summary(print_fn=lambda x: stringlist.append(x))

st.markdown("```\n" + "\n".join(stringlist) + "\n```")
pattern = r"(\S+)\s+(\S+)\s+(\([\d, None]+\))\s+([\d,]+)"
summary_data = []

for line in stringlist[1:-4]:  
    match = re.match(pattern, line)
    if match:
        summary_data.append(match.groups())

st.markdown("### 🎯 ตัวอย่างการทำนายจาก Test Set")
try:
    with st.spinner("🔄 Loading and Training Model"):
        df_test = pd.read_csv(r'Data_set/education_career_bad_model.csv')

    #เตรียมข้อมูล (Data Cleaning)
        df_test_cleaned = df_test.drop(columns=["Student_ID", "Unnamed: 11"], errors="ignore")
        for col in df_test_cleaned.select_dtypes(include=["float64", "int64"]).columns:
            df_test_cleaned[col] = df_test_cleaned[col].fillna(df_test_cleaned[col].mean())
        for col in df_test_cleaned.select_dtypes(include=["object"]).columns:
            df_test_cleaned[col] = df_test_cleaned[col].fillna(df_test_cleaned[col].mode()[0])

    #โหลด Training Columns
        if os.path.exists(COLUMNS_PATH):
            with open(COLUMNS_PATH, "rb") as f:
                train_columns = pickle.load(f)

            for col in train_columns:
                if col not in df_test_cleaned.columns:
                    df_test_cleaned[col] = 0

            df_test_cleaned = df_test_cleaned[train_columns]
        else:
            st.error("❌ ไม่พบไฟล์ train_columns.pkl กรุณา Train โมเดลก่อน")
            st.stop()

    #โหลด Scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            st.success("✅ โหลด Scaler สำเร็จ")
        else:
            st.error("❌ ไม่พบไฟล์ scaler.pkl กรุณา Train โมเดลก่อน")
            st.stop()

    #แปลงข้อมูล
        X_test = df_test_cleaned
        X_test_scaled = scaler.transform(X_test)
        df_test["Starting_Salary"].fillna(df_test["Starting_Salary"].mean(), inplace=True)
        y_test = np.log1p(df_test["Starting_Salary"])
    
        y_pred = model.predict(X_test_scaled)
        
    #Feature , TargetMatrix
        st.write("Feature Matrix")
        st.write(X_test[:10])
        st.write("Target Matrix")
        st.write(y_test[:10])

    #กราฟ Loss & MAE จาก Training
        st.markdown("### 📈 กราฟ Training Loss & MAE")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        ax[0].plot(history["loss"], label="Train Loss")
        ax[0].plot(history["val_loss"], label="Validation Loss")
        ax[0].set_title("Training & Validation Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
    
        ax[1].plot(history["mae"], label="Train MAE")
        ax[1].plot(history["val_mae"], label="Validation MAE")
        ax[1].set_title("Training & Validation MAE")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Mean Absolute Error")
        ax[1].legend()

        st.pyplot(fig)
        st.success("✅ โมเดล Train เสร็จเรียบร้อย! 🚀")
        
        

        
except Exception as e:
    st.error(f"{str(e)}")


