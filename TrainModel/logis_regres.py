import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def train_logistic_model():
    df = pd.read_csv(r'Data_set/netflix_users_bad_model.csv' , nrows=1000)

# ลบค่าที่ Country ผิดปกติ
    df = df[df['Country'].apply(lambda x: isinstance(x, str) and x.isalpha())]

# จัดการ Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] >= 10) & (df['Age'] <= 100)]  # ลบข้อมูลที่มีอายุน้อยกว่า 10 หรือเกิน 100
    df['Age'].fillna(df['Age'].mean(), inplace=True)  # เติมค่าที่หายไปด้วยค่าเฉลี่ย
    df['Age'] = df['Age'].astype(int)

# จัดการ Watch_Time_Hours
    df['Watch_Time_Hours'] = pd.to_numeric(df['Watch_Time_Hours'], errors='coerce')
    df['Watch_Time_Hours'].fillna(df['Watch_Time_Hours'].median(), inplace=True)  # เติมค่าที่หายไปด้วยค่า median

# จัดการ Favorite_Genre
    df = df[df['Favorite_Genre'].apply(lambda x: isinstance(x, str) and not x.isnumeric())]
    df['Favorite_Genre'].replace(['0', '2323'], df['Favorite_Genre'].mode()[0], inplace=True)  # แทนค่าผิดปกติด้วย mode

# แปลงข้อมูลเป็นตัวเลข
    label_encoders = {}
    for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# เลือก Features และ Target / แบ่งข้อมูล Train/Test 
    X = df[['Age', 'Watch_Time_Hours', 'Country', 'Favorite_Genre']]
    y = df['Subscription_Type']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_test, y_test 