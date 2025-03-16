import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from TrainModel.logis_regres import train_logistic_model
from TrainModel.random_forest import load_and_train_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

st.title("🧠 Machine Learning Model")
with st.spinner("🔄 Loading and Training Model"):
    
# เรียกใช้ฟังก์ชันฝึกโมเดล
    model, accuracy, X_test, y_test = load_and_train_model()
    st.success("**Model trained successfully!**")
    st.write("Accuracy: ", accuracy)

    # ทำนายผลกับชุดทดสอบ
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # คำนวณค่า Metric ต่างๆ
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # แสดงผลลัพธ์ของโมเดล
    st.write("🎯 **Model Performance Metrics**")
    metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    }

    df_metrics = pd.DataFrame(metrics_data)
    col1, col2, col3, col4 = st.columns(4)

    # แสดงค่าแต่ละ metric
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1-score", f"{f1:.4f}")

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
# แสดงตารางผลลัพธ์ ของ Dataset เมื่อได้รับการทำนายจากโมเดล
    st.text("Dataset with Predictions")
    df_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Predicted Probability": y_pred_proba
    })
    st.write(df_results)
    

# ---------------------------------------------- Logistic Regression ---------------------------------------------- #
st.subheader("Logistic Regression")

logistic_model, accuracy_lr, X_test_lr, y_test_lr = train_logistic_model()
st.success("**Logistic Regression Model trained successfully!**")
st.write("Model Accuracy: ", accuracy_lr)

# ใช้เฉพาะ Feature
X_feature = X_test_lr[:, 0].reshape(-1, 1)  # ใช้เฉพาะ Age (Feature ที่ 1)

X_range = np.linspace(X_feature.min(), X_feature.max(), 300).reshape(-1, 1)
X_range_full = np.tile(X_test_lr.mean(axis=0), (300, 1))  
X_range_full[:, 0] = X_range[:, 0]  

y_test_prob = logistic_model.predict_proba(X_test_lr)[:, 1]  # คำนวณ probability ของ test data

fig, ax = plt.subplots()
ax.scatter(X_feature, y_test_prob, color='red', label="Actual Data (Predicted Probability)", alpha=0.5)  # จุดข้อมูลจริง
ax.set_xlabel("Feature (Age)")
ax.set_ylabel("Probability")
ax.legend()

st.pyplot(fig)
