import streamlit as st

st.set_page_config(page_title='Perch | property prices in Sri Lanka', layout='wide')

pages = [
    st.Page('app_pages/houses.py', title='Houses', icon=':material/home:', default=True),
    st.Page('app_pages/land.py', title='Land', icon=':material/landscape:'),
]
st.navigation(pages, position='hidden').run()
