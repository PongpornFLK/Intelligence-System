import streamlit as st
import pandas as pd

st.title("Model development guidelines")
st.subheader("แนวทางการพัฒนาด้วย Neural Network แบบ Multilayer Perceptron (MLP)")

st.link_button("🔗 Dataset" , "https://www.kaggle.com/datasets/adilshamim8/education-and-career-success")

# แสดงรายละเอียดของ Dataset
csv_path = "./Data_set/education_career_bad_model.csv"
df = pd.read_csv(csv_path)
st.write(df.head(5))



st.markdown("**:blue[1. การเตรียมข้อมูล :red[(] Data Preparation :red[)]]**" )
st.write("- ลบแถวที่มีค่า NaN ออกจาก DataFrame , ลบแถวที่มีค่าผิดปกติในคอลัมน์ 'Starting_Salary'", unsafe_allow_html=True) 
st.code("""
df = df.dropna()
df = df[(df["Starting_Salary"] > 5000) & (df["Starting_Salary"] < 200000)]
""")
st.write("- เติมค่าหายไป: เติมค่าหายไปในคอลัมน์ที่เป็นตัวเลขด้วยค่าเฉลี่ย และในคอลัมน์ที่เป็นประเภทด้วยค่า Mode", unsafe_allow_html=True) 
st.code("""
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
""")
st.write("- แปลงข้อมูล Categorical เป็น One-Hot Encoding ", unsafe_allow_html=True) 
st.code("""
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
""")
st.write("- บันทึกรายชื่อ Features เพื่อใช้ตอนทดสอบ ", unsafe_allow_html=True) 
st.code("""
feature_columns = df.drop(columns=["Starting_Salary"]).columns.tolist()
with open(COLUMNS_PATH, "wb") as f:
    pickle.dump(feature_columns, f)
""")
st.write("- แยก Features (X) และ Target (y) และ ลบแถวที่มีค่า NaN ใน X และ y พร้อมกัน", unsafe_allow_html=True) 
st.code("""
df_clean = pd.concat([X, y], axis=1).dropna() 
X_clean = df_clean[feature_columns]  
y_clean = df_clean["Starting_Salary"]  
""")
st.write("- ทำ Feature Scaling: ทำการ Normalize ข้อมูล ", unsafe_allow_html=True) 
st.code("""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)  # ใช้ X_clean ที่ทำความสะอาดแล้ว

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
""")


st.markdown("**:blue[2. ทฤษฎีของอัลกอริทึมที่พัฒนา]**")
st.write("- Multilayer Perceptron (MLP) เป็น Neural Network ที่ประกอบด้วยหลายชั้นของ Fully Connected Layers พร้อมฟังก์ชัน Activation เช่น ReLU")
st.write(":green-background[:green[**ข้อดีของ MLP:**]] : <br>-- รองรับความซับซ้อนของข้อมูลได้ดี <br>-- สามารถใช้ฟังก์ชัน Activation ต่างๆ เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น <br>-- ใช้เทคนิค Regularization เพื่อลด Overfitting", unsafe_allow_html=True)

st.markdown("**:blue[3. การพัฒนาโมเดล Neural Network]**")
st.write("- สร้างและ Train โมเดล MLP Neural Network", unsafe_allow_html=True)
st.code("""
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

    keras.layers.Dense(1, activation='sigmoid')  # เนื่องจากเป็นปัญหาการจำแนกประเภท
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
""")

st.write("- ประเมินผลโมเดลโดยใช้ Accuracy, MSE, MAE", unsafe_allow_html=True)
st.code("""
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # แปลงผลลัพธ์เป็น 0 หรือ 1

accuracy = np.mean(y_pred == y_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

st.write(f"🔹 **Accuracy:** {accuracy:.4f}")
st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"🔹 **Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.4f}")
st.write(f"🔹 **R² Score (R2):** {r2:.4f}")
st.write(f"🔹 **Mean Absolute Percentage Error (MAPE):** {mape:.4f}")
""")

st.write("**ดังนั้น** Neural Network ที่พัฒนาขึ้นสามารถใช้ในการจำแนกประเภทผู้ใช้ Netflix ได้อย่างมีประสิทธิภาพ พร้อมทั้งใช้เทคนิค Regularization เพื่อลด Overfitting และเพิ่มความแม่นยำ", unsafe_allow_html=True)



