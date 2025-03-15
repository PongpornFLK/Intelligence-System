import streamlit as st
import numpy as np
from TrainModel.random_forest import load_and_train_model
from TrainModel.logis_regres import train_logistic_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc



st.title("🧠 Model")
st.write("Loading and training the model...")

# เรียกใช้ฟังก์ชันฝึกโมเดล
model, accuracy, X_test, y_test = load_and_train_model()
st.success("**Model trained successfully!**")
st.write("Model Accuracy : ", accuracy)

# ทำนายผลกับชุดทดสอบ
y_pred = model.predict(X_test)

# คำนวณ confusion matrix
cm = confusion_matrix(y_test, y_pred)

# สร้างกราฟ heatmap สำหรับ confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# แสดงความสำคัญของแต่ละ feature
st.subheader("Feature Importance")
features = ['Age', 'Watch_Time_Hours', 'Country', 'Favorite_Genre']
importances = model.feature_importances_

feat, ax = plt.subplots()
ax.barh(features, importances, color='skyblue')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance in Random Forest')

st.pyplot(feat)

# แสดงข้อมูลแบบตาราง
st.subheader("Sample of Processed Test Data")
test_data = pd.DataFrame(X_test, columns=features)
test_data['Actual Subscription Type'] = y_test.values
test_data['Predicted Subscription Type'] = y_pred

st.dataframe(test_data.head(10))



# ---------------------------------------------- Logistic Regression ---------------------------------------------- #
# ใน Streamlit
st.subheader("Logistic Regression")

# โหลดและฝึกโมเดล Logistic Regression
logistic_model, accuracy_lr, X_test_lr, y_test_lr = train_logistic_model()  # ใช้ 4 ตัวแปร

st.success("**Logistic Regression Model trained successfully!**")

# คำนวณ ROC curve และ AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_lr)
roc_auc = auc(fpr, tpr)

# สร้างกราฟ ROC Curve
st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')

st.pyplot(fig)

# สร้าง Probability Curve หรือ Graph เส้น
st.subheader("Probability Curve")
fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(y_test_lr)), y_test_lr, marker='o', linestyle='-', color='green', label='Predicted Probabilities')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Probability')
ax2.set_title('Logistic Regression Probability Curve')
ax2.legend()

st.pyplot(fig2)

