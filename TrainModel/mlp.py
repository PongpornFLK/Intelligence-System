import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import pickle

def load_neural_model():
    # 🚀 1. โหลด Dataset
    df = pd.read_csv(r'Data_set/education_career_realmodel.csv')

    # ✅ 2. ลบ Outliers ก่อนทำ Data Cleaning
    df_cleaned = df[(df["Starting_Salary"] > 5000) & (df["Starting_Salary"] < 200000)]
    print(f"✅ ขนาดข้อมูลหลังลบ Outliers: {df_cleaned.shape}")

    # ✅ 3. เติมค่าหายไป
    for col in df_cleaned.select_dtypes(include=["float64", "int64"]).columns:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    for col in df_cleaned.select_dtypes(include=["object"]).columns:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    # ✅ 4. แปลง Categorical Data เป็นตัวเลข
    label_encoders = {}
    for col in df_cleaned.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoders[col] = le

    # ✅ 5. แยก Features (X) และ Target (y)
    X = df_cleaned.drop(columns=["Starting_Salary"])
    y = np.log1p(df_cleaned["Starting_Salary"])  # ✅ Log Transform Target

    # ✅ 6. ทำ Feature Scaling และบันทึก Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ✅ 7. แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 🚀 8. สร้าง MLP Model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
        keras.layers.Dense(32, activation='relu'),  
        keras.layers.Dense(16, activation='relu'),  
        keras.layers.Dense(1)  # Output Layer สำหรับ Regression
    ])

    # 🚀 9. คอมไพล์โมเดล
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mae'])

    # 🚀 10. Train Model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # ✅ 11. บันทึกโมเดลและค่าประวัติการ Train
    model.save("load_model.h5")
    with open("load_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("\n✅ โมเดลถูกบันทึกแล้วเป็น load_model.h5 และ load_history.pkl")
