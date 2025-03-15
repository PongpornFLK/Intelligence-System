import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_train_model():
    # โหลดข้อมูล
    netflix_df = pd.read_csv(r'Data_set\netflix_users_bad_model.csv')
    
    # 1. แก้ค่าที่ผิดปกติในคอลัมน์ 'Country'
    netflix_df = netflix_df[~netflix_df['Country'].isin([0, '123214', '()&*_)+'])]  # ลบแถวที่ Country มีค่าผิดปกติ
    
    # 2. จัดการคอลัมน์ 'Age'
    netflix_df['Age'] = pd.to_numeric(netflix_df['Age'], errors='coerce')  # แปลงเป็นตัวเลข ถ้าเป็นตัวอักษรจะกลายเป็น NaN
    netflix_df = netflix_df[(netflix_df['Age'] >= 10) & (netflix_df['Age'] <= 100)]  # ลบอายุที่ไม่ถูกต้อง
    netflix_df['Age'].fillna(netflix_df['Age'].mean(), inplace=True)  # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    netflix_df['Age'] = netflix_df['Age'].astype(int)  # เปลี่ยนอายุเป็นจำนวนเต็ม
    
    # 3. จัดการคอลัมน์ 'Watch_Time_Hours'
    netflix_df['Watch_Time_Hours'] = pd.to_numeric(netflix_df['Watch_Time_Hours'], errors='coerce')  # แปลงเป็นตัวเลข
    netflix_df = netflix_df[~netflix_df['Watch_Time_Hours'].isna()]  # ลบแถวที่ Watch_Time_Hours เป็น NaN
    netflix_df['Watch_Time_Hours'].fillna(netflix_df['Watch_Time_Hours'].median(), inplace=True)  # เติมค่าที่หายไปด้วยค่ามัธยฐาน
    
    # 4. แก้ไข Favorite_Genre ที่ผิดปกติ
    invalid_genres = ['0', '2323']
    netflix_df['Favorite_Genre'] = netflix_df['Favorite_Genre'].replace(invalid_genres, np.nan)  # แทนที่ค่าผิดปกติด้วย NaN
    netflix_df['Favorite_Genre'].fillna(netflix_df['Favorite_Genre'].mode()[0], inplace=True)  # เติมค่าผิดปกติด้วยค่า Mode (ค่าที่พบมากที่สุด)
    
    # 5. แปลงข้อมูล Categorical ให้เป็นตัวเลข
    label_encoders = {}
    for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
        le = LabelEncoder()
        netflix_df[col] = le.fit_transform(netflix_df[col])
        label_encoders[col] = le
    
    # เลือก Feature และ Target
    X = netflix_df[['Age', 'Watch_Time_Hours', 'Country', 'Favorite_Genre']]
    y = netflix_df['Subscription_Type']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # สร้างและ Train โมเดล Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # ประเมินโมเดลด้วย Accuracy
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    return rf, accuracy, X_test, y_test

# สำหรับทดสอบแบบ Standalone
if __name__ == '__main__':
    model, acc, _, _ = load_and_train_model()
    print("Model accuracy:", acc)
