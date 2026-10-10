import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
GREEN, SAFFRON = '#0F5C50', '#F2B84B'

st.set_page_config(page_title='Perch | house prices in Sri Lanka', layout='wide')


@st.cache_resource
def load_model():
    with open(ROOT / 'model' / 'house_price_model.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_json(name):
    with open(ROOT / 'model' / name) as f:
        return json.load(f)


@st.cache_data
def load_houses():
    return pd.read_csv(ROOT / 'data' / 'processed' / 'houses_clean.csv')


model = load_model()
meta = load_json('model_meta.json')
sample = load_json('holdout_sample.json')
houses = load_houses()


def short(price):
    return f'Rs {price / 1e6:,.1f} M'


# ---------- state ----------
districts = sorted(meta['locations'])
DEFAULTS = {
    'district': 'Colombo', 'location': meta['locations']['Colombo'][0],
    'bedrooms': 3, 'bathrooms': 2, 'house_size_sqft': 2000, 'land_size_perches': 10.0,
    'membership_level': 'free', 'is_member': 'No', 'is_verified': 'No',
    'month': MONTHS[date.today().month - 1],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_sample():
    f = sample['features']
    st.session_state.update({
        'district': f['district'], 'location': f['location'],
        'bedrooms': int(f['bedrooms']), 'bathrooms': int(f['bathrooms']),
        'house_size_sqft': int(f['house_size_sqft']), 'land_size_perches': float(f['land_size_perches']),
        'membership_level': f['membership_level'],
        'is_member': 'Yes' if f['is_member'] else 'No',
        'is_verified': 'Yes' if f['is_verified'] else 'No',
        'month': MONTHS[int(f['posted_month']) - 1],
        'sample_loaded': True,
    })


def reset_flag():
    st.session_state['sample_loaded'] = False


def district_changed():
    st.session_state['location'] = meta['locations'][st.session_state['district']][0]
    reset_flag()


# ---------- header ----------
st.title('Perch')
st.write('Describe a house and get a price estimate.')

known = set((d, l) for d in meta['locations'] for l in meta['locations'][d])
left, right = st.columns([3, 2], gap='large')

with left:
    st.subheader('Describe the house')
    c1, c2 = st.columns(2)
    district = c1.selectbox('District', districts, key='district', on_change=district_changed)
    location = c2.selectbox('Location', meta['locations'][district], key='location', on_change=reset_flag)

    c3, c4 = st.columns(2)
    bedrooms = c3.number_input('Bedrooms', 1, 10, key='bedrooms', on_change=reset_flag)
    bathrooms = c4.number_input('Bathrooms', 1, 10, key='bathrooms', on_change=reset_flag)

    c5, c6 = st.columns(2)
    house_size = c5.number_input('House size (sqft)', 200, 20000, step=50, key='house_size_sqft', on_change=reset_flag)
    land_size = c6.number_input('Land size (perches)', 1.0, 500.0, step=0.5, key='land_size_perches', on_change=reset_flag)

    with st.expander('Seller details'):
        c7, c8, c9, c10 = st.columns(4)
        membership = c7.selectbox('Membership', meta['membership_levels'], key='membership_level', on_change=reset_flag)
        is_member = c8.selectbox('Member', ['No', 'Yes'], key='is_member', on_change=reset_flag)
        is_verified = c9.selectbox('Verified', ['No', 'Yes'], key='is_verified', on_change=reset_flag)
        month = c10.selectbox('Posted in', MONTHS, key='month', on_change=reset_flag)

    st.button('Try the held out sample listing', on_click=load_sample)


def make_row(**changes):
    row = {
        'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'house_size_sqft': house_size, 'land_size_perches': land_size,
        'posted_month': MONTHS.index(month) + 1,
        'is_verified': int(is_verified == 'Yes'),
        'location': location, 'district': district,
        'membership_level': membership, 'is_member': float(is_member == 'Yes'),
    }
    row.update(changes)
    return row


price = float(model.predict(pd.DataFrame([make_row()]))[0])

# about 68% of listings fall inside +/- one test rmsle on the log scale
spread = np.exp(meta['test_rmsle'])
low, high = price / spread, price * spread

with right:
    st.metric('Estimated asking price', short(price))
    st.caption(f'Rs {price:,.0f}')
    st.write(f'Likely range: {short(low)} to {short(high)}')

    quantiles = meta['price_quantiles'].get(district)
    if quantiles:
        pct = float(np.interp(price, quantiles, np.arange(101)))
        st.write(f'Priced above about {pct:.0f}% of house ads in {district}.')

    t1, t2 = st.columns(2)
    t1.metric('Per sqft', f'Rs {price / house_size:,.0f}')
    t2.metric('Per perch of land', short(price / land_size))

    if st.session_state.get('sample_loaded'):
        actual = sample['actual_price']
        err = abs(price - actual) / actual * 100
        st.info(f'Held out listing. It was never used for training. The real asking price was {short(actual)}, so the estimate is off by {err:.1f}%.')

    # the same house in other busy areas
    busy = (houses.groupby(['location', 'district']).size().rename('ads').reset_index()
            .query('ads >= 100').sort_values('ads', ascending=False).head(12))
    busy = busy[[(d, l) in known for d, l in zip(busy['district'], busy['location'])]]
    what_if = pd.DataFrame([make_row(location=l, district=d) for l, d in zip(busy['location'], busy['district'])])
    compare = pd.Series(model.predict(what_if) / 1e6, index=busy['location'].values, name='Rs million')

    st.subheader('The same house elsewhere')
    st.bar_chart(compare.sort_values(), horizontal=True, color=GREEN, height=330)

    # price against house size for this location
    sizes = np.arange(500, 6001, 250)
    curve = pd.DataFrame([make_row(house_size_sqft=int(s)) for s in sizes])
    st.subheader('Price as the house gets bigger')
    st.line_chart(pd.Series(model.predict(curve) / 1e6, index=sizes, name='Rs million'), color=SAFFRON, height=240)

    st.caption(f'Estimates are based on asking prices from ads, not final sale prices. Model: {meta["model"]}.')
