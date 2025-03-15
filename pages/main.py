import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# แสดง Title
st.title("📝 Overview")


# # โหลด Dataset
# csv_path = "./Data_set/netflix_users_bad_model.csv"
# df = pd.read_csv(csv_path)

# # แสดง DataFrame
# st.subheader("🔍 Detail")
# st.write(df.head(20))  # แสดง 10 แถวแรก


st.markdown("**Resource of Dataset** : ")
col1, col2 = st.columns(2)
with col1:
    st.text("Education & Career Success" )
    st.link_button("🔗 Click Link" , "https://www.kaggle.com/datasets/adilshamim8/education-and-career-success")
    
    

with col2:
    st.text("Netflix Users Database")
    st.link_button("🔗 Click Link" , "https://www.kaggle.com/datasets/smayanj/netflix-users-database")

    


