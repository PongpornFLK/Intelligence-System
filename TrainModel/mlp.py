import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import os

def load_neural_model():
    # 🚀 1. โหลด Dataset
    df = pd.read_csv(r'Data_set/education_career_bad_model.csv')
    
    # ✅ 2. ลบค่าผิดปกติ (Outliers) ใน Starting_Salary
    df = df[(df["Starting_Salary"] > 5000) & (df["Starting_Salary"] < 200000)]
    print(f"✅ ขนาดข้อมูลหลังลบ Outliers: {df.shape}")
    
    # ✅ 3. เติมค่าหายไป
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)  # ใช้ mode สำหรับ Categorical
    
    # ✅ 4. แปลง Categorical Data เป็นตัวเลขด้วย One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # ✅ 5. แยก Features (X) และ Target (y)
    X = df.drop(columns=["Starting_Salary"])
    y = np.log1p(df["Starting_Salary"])  # ✅ ใช้ Log Transform ลดความเบ้ของข้อมูล
    
    # ✅ 6. ทำ Feature Scaling และบันทึก Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ensure the directory exists
    os.makedirs("TrainModel", exist_ok=True)
    
    with open("TrainModel/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # ✅ 7. แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 🚀 8. สร้าง MLP Model ที่ปรับให้เหมาะสม
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # เพิ่มจำนวน Neurons
        keras.layers.Dropout(0.2),  # ใช้ Dropout ป้องกัน Overfitting
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Output Layer สำหรับ Regression
    ])
    
    # 🚀 9. คอมไพล์โมเดล
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    
    # 🚀 10. Train Model พร้อม Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        epochs=100, batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping], verbose=1)
    
    # ✅ 11. บันทึกโมเดลและค่าประวัติการ Train
    model.save("load_model.h5")
    with open("load_history.pkl", "wb") as f:
        pickle.dump(history.history, f)