import streamlit as st

st.title("Model development guidelines")
# -------------------------------- Random Forest --------------------------------
st.subheader("แนวทางการพัฒนาโมเดลแบบ Random Forest")
st.markdown("**:blue[1. การเตรียมข้อมูล :red[(] Data Preparation :red[)]]**" )
st.write("- เริ่มจากการนำเข้า Dataset ที่จะใช้พัฒนาโมเดล ในที่นี้ใช้ไฟล์ netflix_users_bad_model.csv โดยมีปัญหาข้อมูลไม่สมบูรณ์หรือมีข้อผิดพลาด เพื่อนำมาวิเคราะห์", unsafe_allow_html=True) 
st.code("""
netflix_df = pd.read_csv(r'Data_set/netflix_users_bad_model.csv', nrows=1000)
""")
st.write("- จัดการค่าผิดปกติในคอลัมน์ 'Country': ลบแถวที่มีค่าผิดปกติในคอลัมน์ 'Country'", unsafe_allow_html=True) 
st.code("""
netflix_df = netflix_df[~netflix_df['Country'].isin([0, '123214', '()&*_)+'])]""")
st.write("- จัดการคอลัมน์ 'Age': แปลงค่าในคอลัมน์ 'Age' เป็นตัวเลข ลบค่าที่ไม่ถูกต้อง และเติมค่าที่หายไปด้วยค่าเฉลี่ย", unsafe_allow_html=True) 
st.code("""
netflix_df['Age'] = pd.to_numeric(netflix_df['Age'], errors='coerce')
netflix_df = netflix_df[(netflix_df['Age'] >= 10) & (netflix_df['Age'] <= 100)]
netflix_df['Age'].fillna(netflix_df['Age'].mean(), inplace=True)
netflix_df['Age'] = netflix_df['Age'].astype(int)
""")
st.write("- จัดการคอลัมน์ 'Watch_Time_Hours': แปลงค่าในคอลัมน์ 'Watch_Time_Hours' เป็นตัวเลข และเติมค่าที่หายไปด้วยค่ามัธยฐาน และ แก้ไขค่าผิดปกติในคอลัมน์ 'Favorite_Genre': แทนที่ค่าผิดปกติด้วย NaN ", unsafe_allow_html=True) 
st.code("""
netflix_df['Watch_Time_Hours'] = pd.to_numeric(netflix_df['Watch_Time_Hours'], errors='coerce')
netflix_df = netflix_df[~netflix_df['Watch_Time_Hours'].isna()]
netflix_df['Watch_Time_Hours'].fillna(netflix_df['Watch_Time_Hours'].median(), inplace=True)

netflix_df['Favorite_Genre'] = netflix_df['Favorite_Genre'].replace(invalid_genres, np.nan)
netflix_df['Favorite_Genre'].fillna(netflix_df['Favorite_Genre'].mode()[0], inplace=True)
""")
st.write("- แปลงข้อมูล Categorical ให้เป็นตัวเลข", unsafe_allow_html=True) 
st.code("""
label_encoders = {}
for col in ['Country', 'Subscription_Type', 'Favorite_Genre']:
    le = LabelEncoder()
    netflix_df[col] = le.fit_transform(netflix_df[col])
    label_encoders[col] = le
""")

st.write("- เลือก Feature และ Target: เลือกคอลัมน์ที่ใช้เป็น Feature และ Target แล้วใช้ StandardScaler เพื่อทำการ Normalize ข้อมูล:", unsafe_allow_html=True) 
st.code("""
X = netflix_df[['Age', 'Watch_Time_Hours', 'Country', 'Favorite_Genre']]
y = netflix_df['Subscription_Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
""")


st.markdown("**:blue[2. ทฤษฎีของอัลกอริทึมที่พัฒนา]**" )
st.write("- Random Forest เป็นอัลกอริทึมที่ใช้สำหรับการจำแนกประเภท (classification) และการถดถอย (regression) ด้วยการสร้างหลายๆ Decision Trees และใช้การโหวตเสียงข้างมาก (majority voting) ในการจำแนกประเภท หรือเฉลี่ยผลลัพธ์", unsafe_allow_html=True)
st.write(":green-background[:green[**ข้อดีของ Random Forest:**]] : <br>-- ลดการ Overfitting: การใช้หลายๆ ต้นไม้ช่วยลดการ Overfitting ที่อาจเกิดขึ้นจากการใช้ Decision Tree เพียงต้นเดียว <br>-- ความแม่นยำสูง: Random Forest มักให้ผลลัพธ์ที่มีความแม่นยำสูงเนื่องจากการรวมผลลัพธ์จากหลายๆ ต้นไม้ <br>-- ทนทานต่อการเปลี่ยนแปลงของข้อมูล: Random Forest มีความทนทานต่อการเปลี่ยนแปลงของข้อมูลและการสุ่มตัวอย่าง", unsafe_allow_html=True) 

st.markdown("**:blue[3. การพัฒนาโมเดล Random Forest]**" )
st.write("- สร้างและ Train โมเดล Random Forest", unsafe_allow_html=True)
st.code(f"""
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
""")
st.write("- ประเมินโมเดลด้วย Accuracy แล้วคืนค่าผลลัพธ์", unsafe_allow_html=True)
st.code("""
accuracy = accuracy_score(y_test, rf.predict(X_test))

return rf, accuracy, X_test, y_test
""")

st.write("- แบ่งข้อมูล Train/Test", unsafe_allow_html=True) 
st.code("""
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
""")
st.write("**ดังนั้น** Random Forest นี้ช่วยให้สามารถจำแนกประเภทของผู้ใช้ Netflix ได้อย่างมีประสิทธิภาพ โดยใช้ข้อมูลที่เตรียมไว้อย่างเหมาะสมและการประเมินผลลัพธ์ด้วยความแม่นยำ", unsafe_allow_html=True)

# -------------------------------- Logistic Regression --------------------------------
st.subheader("แนวทางการพัฒนาโมเดลแบบ Logistic Regression")
st.markdown("**:blue[1. การเตรียมข้อมูล :red[(] Data Preparation :red[)]]**" )


st.markdown("**:blue[2. ทฤษฎีของอัลกอริทึมที่พัฒนา]**" )


st.markdown("**:blue[3. การพัฒนาโมเดล Random Forest]**" )







