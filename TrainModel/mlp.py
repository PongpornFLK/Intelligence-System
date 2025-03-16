import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import os

# 📌 กำหนด Paths ของไฟล์ที่ต้องบันทึก
MODEL_DIR = "TrainModel"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "train_columns.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "load_model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "load_history.pkl")

# ✅ ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(MODEL_DIR, exist_ok=True)

def load_neural_model():

    df = pd.read_csv(r'Data_set/education_career__model.csv' , nrows=1000)
    
    df = df.dropna()  # ลบแถวที่มีค่า NaN ทิ้ง    

    df = df[(df["Starting_Salary"] > 5000) & (df["Starting_Salary"] < 200000)]

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df.dropna(inplace=True)

    feature_columns = df.drop(columns=["Starting_Salary"]).columns.tolist()
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(feature_columns, f)

    X = df[feature_columns]
    y = np.log1p(df["Starting_Salary"])  # ใช้ Log Transform ลดความเบ้ของข้อมูล
    
    # เช็คว่า X และ y มีค่า NaN หรือไม่
    if X.isnull().any().any() or y.isnull().any():
        print("❌ พบ NaN ใน X หรือ y")
        return
    # ลบแถวที่มี NaN ในทั้ง X และ y พร้อมกัน
    df_clean = pd.concat([X, y], axis=1).dropna()  # รวม X และ y แล้วลบแถวที่มี NaN
    X_clean = df_clean[feature_columns]  # X หลังจากลบ NaN
    y_clean = df_clean["Starting_Salary"]  # y หลังจากลบ NaN
    
    # ตรวจสอบขนาดของ X และ y
    if X_clean.shape[0] != y_clean.shape[0]:
        print(f"❌ ขนาดของ X และ y ไม่เท่ากัน! X: {X_clean.shape[0]}, y: {y_clean.shape[0]}")
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)  # ใช้ X_clean ที่ทำความสะอาดแล้ว

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(1)  # Output Layer สำหรับ Regression
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(X_train, y_train,
                        epochs=150, batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    model.save(MODEL_PATH)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

