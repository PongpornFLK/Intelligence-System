import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from TrainModel.random_forest import load_and_train_model
from TrainModel.logis_regres import train_logistic_model
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

st.title("🧠 Machine Learning Model")
with st.spinner("🔄 Loading and Training Model"):
    # เรียกใช้ฟังก์ชันฝึกโมเดล
    model, accuracy, X_test, y_test = load_and_train_model()
    st.success("**Model trained successfully!**")
    st.write("Model Accuracy: ", accuracy)

    # ทำนายผลกับชุดทดสอบ
    y_pred = model.predict(X_test)

    # คำนวณ confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # สร้างกราฟ heatmap สำหรับ confusion matrix
    st.subheader("Random Forest")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)


# ---------------------------------------------- Logistic Regression ---------------------------------------------- #
# ใน Streamlit
st.subheader("Logistic Regression")

# โหลดและฝึกโมเดล Logistic Regression
logistic_model, accuracy_lr, X_test_lr, y_test_lr = train_logistic_model()

st.success("**Logistic Regression Model trained successfully!**")
st.write("Model Accuracy: ", accuracy_lr)

# ใช้เฉพาะ Feature เดียว (เช่น อายุ)
X_feature = X_test_lr[:, 0].reshape(-1, 1)  # ใช้เฉพาะ Age (Feature ที่ 1)

# **สร้างค่า X_range** สำหรับพล็อตกราฟเส้น S
X_range = np.linspace(X_feature.min(), X_feature.max(), 300).reshape(-1, 1)

# **สร้าง X_range ที่มี 4 ฟีเจอร์** (ใช้ค่าเฉลี่ยจาก X_test_lr สำหรับฟีเจอร์อื่น)
X_range_full = np.tile(X_test_lr.mean(axis=0), (300, 1))  # ทำค่าเฉลี่ยของ X_test_lr
X_range_full[:, 0] = X_range[:, 0]  # อัปเดตค่าเฉพาะฟีเจอร์ Age

# **คำนวณค่าความน่าจะเป็น**
y_test_prob = logistic_model.predict_proba(X_test_lr)[:, 1]  # คำนวณ probability ของ test data

fig, ax = plt.subplots()
ax.scatter(X_feature, y_test_prob, color='red', label="Actual Data (Predicted Probability)", alpha=0.5)  # จุดข้อมูลจริง
ax.set_xlabel("Feature (Age)")
ax.set_ylabel("Probability")
ax.legend()

st.pyplot(fig)
