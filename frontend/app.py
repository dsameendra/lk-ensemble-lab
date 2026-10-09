from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

st.set_page_config(page_title='Perch | house prices in Sri Lanka', layout='wide')


@st.cache_data
def load_houses():
    return pd.read_csv(ROOT / 'data' / 'processed' / 'houses_clean.csv')


houses = load_houses()

st.title('Perch')
st.write('Describe a house and get a price estimate.')

st.subheader('Describe the house')
c1, c2 = st.columns(2)
district = c1.selectbox('District', sorted(houses['district'].unique()))
places = sorted(houses.loc[houses['district'] == district, 'location'].unique())
location = c2.selectbox('Location', places)

c3, c4 = st.columns(2)
bedrooms = c3.number_input('Bedrooms', 1, 10, 3)
bathrooms = c4.number_input('Bathrooms', 1, 10, 2)

c5, c6 = st.columns(2)
house_size = c5.number_input('House size (sqft)', 200, 20000, 2000, step=50)
land_size = c6.number_input('Land size (perches)', 1.0, 500.0, 10.0, step=0.5)

with st.expander('Seller details'):
    c7, c8, c9, c10 = st.columns(4)
    membership = c7.selectbox('Membership', sorted(houses['membership_level'].dropna().unique()))
    is_member = c8.selectbox('Member', ['No', 'Yes'])
    is_verified = c9.selectbox('Verified', ['No', 'Yes'])
    month = c10.selectbox('Posted in', MONTHS)

st.write(f'{bedrooms} bedrooms, {bathrooms} bathrooms, {house_size} sqft house on {land_size} perches in {location}, {district}.')
