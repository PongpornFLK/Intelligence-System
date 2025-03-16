import streamlit as st

main_page = st.Page(page = "pages/overview.py"  , title = "Pongporn Yampradit", icon = ":material/play_arrow:", default=True)
guideline_machine = st.Page(page = "pages/guideline_machine.py"  , title = "Guideline Machine Learning", icon = ":material/play_arrow:")
guideline_neural = st.Page(page = "pages/guideline_neural.py"  , title = "Guideline Neural Network", icon = ":material/play_arrow:")
machine_page = st.Page(page = "pages/machine.py"  , title = "Machine Model", icon = ":material/play_arrow:")
neural_page = st.Page(page = "pages/neural.py"  , title = "Neural Model", icon = ":material/play_arrow:")

pg = st.navigation({ "🧑🏻‍💻 Student Info" : [main_page] , 
                     "📌 Model" : [machine_page, neural_page] ,
                     "📌 Guideline / Reporting" : [guideline_machine,guideline_neural] ,
                  })
pg.run() 
