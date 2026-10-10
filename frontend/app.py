import json
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# ads in the data run from sep 2022 to apr 2023, the first and last months are too thin to use
TIMELINE = [(2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2), (2023, 3)]
TIMELINE_LABELS = [f'{MONTHS[m - 1]} {y}' for y, m in TIMELINE]
GREEN, SAFFRON = '#0F5C50', '#F2B84B'

st.set_page_config(page_title='Perch | house prices in Sri Lanka', layout='wide')

# ---------- styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&family=Sora:wght@600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Manrope', sans-serif; }
span[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #F3F6F5; }
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 0 3rem 0; max-width: 100%; }

.hero { background: #0F3D36; padding: 1.8rem 8vw 5.5rem 8vw; color: #fff; }
.hero .brand { display: flex; align-items: center; gap: .7rem; }
.hero .logo { width: 38px; height: 38px; border-radius: 10px; background: #F2B84B; color: #0F3D36;
              display: grid; place-items: center; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.3rem; }
.hero .name { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; letter-spacing: .5px; }
.hero h1 { font-family: 'Sora', sans-serif; font-size: 2.2rem; line-height: 1.15; margin: 1.2rem 0 .6rem 0; max-width: 40rem; color: #fff; text-wrap: balance; }
.hero p { color: #BFD6D0; font-size: 1.05rem; max-width: 38rem; line-height: 1.55; margin: 0; }

.body { margin: -4rem auto 0 auto; padding: 0 8vw; }
[data-testid="stTabs"] { padding: 0 8vw; }
[class*="st-key-card"], .card {
    background: #fff; border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(16, 35, 31, .10);
}
.card h3, [class*="st-key-card"] h3 { font-family: 'Sora', sans-serif; font-size: 1.1rem; margin: 0 0 .4rem 0; }

.est { border-top: 5px solid #F2B84B; position: relative; overflow: hidden; }
.label { font-size: .75rem; font-weight: 700; color: #4C625D; text-transform: uppercase; letter-spacing: 1.5px; }
.big { font-family: 'Sora', sans-serif; font-size: 2.6rem; font-weight: 700; color: #10231F; margin: .3rem 0 0 0; }
.sub { color: #4C625D; font-size: .95rem; }
.range { margin-top: .8rem; padding: .7rem .9rem; background: #EAF3F0; border-radius: 10px; font-size: .9rem; }

.bar { position: relative; height: 12px; border-radius: 6px; margin: 1rem 0 .5rem 0;
       background: linear-gradient(90deg, #CFE5DF, #0F5C50); }
.bar .pin { position: absolute; top: -5px; left: var(--pos); width: 4px; height: 22px; background: #F2B84B; border-radius: 2px;
            animation: slide .6s cubic-bezier(.16, 1, .3, 1) both; }
.bar-labels { display: flex; justify-content: space-between; font-size: .75rem; color: #4C625D; }

.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.tile { background: #fff; border-radius: 14px; padding: 1rem; box-shadow: 0 4px 16px rgba(16, 35, 31, .08); }
.tile .v { font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 700; }
.note { font-size: .8rem; color: #4C625D; line-height: 1.5; }
.check { margin-top: .8rem; padding: .7rem .9rem; border-radius: 10px; background: #FFF6E0; font-size: .9rem; }

.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin-bottom: 1rem; }
.kpi { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(16, 35, 31, .08); }
.kpi .v { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; }
.kpi .s { font-size: .8rem; color: #4C625D; }

button[data-baseweb="tab"] { font-weight: 700; font-size: 1rem; }
div[data-testid="stButton"] button { border-radius: 12px; height: 2.8rem; font-weight: 700; border: 1px solid #C9D6D2; }
div[data-testid="stButton"] button:hover { border-color: #0F5C50; color: #0F5C50; }

/* numbers line up and the browser bits follow the palette */
.big, .sub, .tile .v, .kpi .v, .range, .bar-labels { font-variant-numeric: tabular-nums; }
::selection { background: #F2B84B; color: #10231F; }
:focus-visible { outline: 3px solid #F2B84B; outline-offset: 2px; }

/* the result card stays in view while the form is edited */
div[data-testid="stColumn"]:has(.st-key-result_sticky) { align-self: stretch; }
[data-testid="stLayoutWrapper"]:has(> .st-key-result_sticky) { position: sticky; top: 1rem; }

/* price counts from the last estimate to the new one */
@property --n { syntax: '<integer>'; inherits: false; initial-value: 0; }
.count { --n: var(--to); counter-reset: whole round(down, calc(var(--n) / 10)) tenth mod(var(--n), 10);
         animation: tick .8s cubic-bezier(.16, 1, .3, 1) both; }
.count::before { content: 'Rs ' counter(whole) '.' counter(tenth) ' M'; }
@keyframes tick { from { --n: var(--from); } }
@keyframes slide { from { left: var(--pin-from); } }
.reveal { animation: rise .4s cubic-bezier(.16, 1, .3, 1) both; }
@keyframes rise { from { opacity: .4; transform: translateY(8px); } }

/* thin saffron line while the page recalculates */
[data-stale="true"] .est::after { content: ''; position: absolute; top: 0; left: 0; height: 3px; width: 35%;
                                  background: #F2B84B; animation: sweep 1s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(300%); } }

.price-bar { display: none; }

@media (max-width: 800px) {
    .hero { padding: 1.2rem 1.2rem 4.5rem 1.2rem; }
    .hero h1 { font-size: 1.6rem; }
    .body, [data-testid="stTabs"] { padding: 0 1rem; }
    .big { font-size: 2rem; }
    .kpis { grid-template-columns: 1fr 1fr; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-result_sticky) { position: static; }
    .block-container { padding-bottom: 5rem; }
    [class*="st-key-card_map"] [data-testid="stDeckGlJsonChart"] { height: 260px !important; }
    .price-bar { display: flex; justify-content: space-between; align-items: center; position: fixed; left: 0; right: 0; bottom: 0;
                 z-index: 1000; padding: .8rem 1.2rem; background: #0F3D36; color: #BFD6D0; font-size: .85rem;
                 box-shadow: 0 -6px 20px rgba(16, 35, 31, .25); font-variant-numeric: tabular-nums; }
    .price-bar b { font-family: 'Sora', sans-serif; font-size: 1.3rem; color: #F2B84B; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ---------- loading ----------
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
    houses = pd.read_csv(ROOT / 'data' / 'processed' / 'houses_clean.csv')
    coords = pd.read_csv(ROOT / 'data' / 'processed' / 'location_coords.csv')
    houses = houses.merge(coords[['location', 'latitude', 'longitude']], on='location', how='left')
    houses['price_per_sqft'] = houses['price'] / houses['house_size_sqft']
    return houses


try:
    model = load_model()
    meta = load_json('model_meta.json')
    sample = load_json('holdout_sample.json')
    houses = load_houses()
except FileNotFoundError:
    st.error('The model or data files are missing. Run the notebooks first, they save everything the app needs.')
    st.stop()


def short(price):
    return f'Rs {price / 1e6:,.1f} M'


def area_stats(data):
    # one row per location with the numbers shown on the maps
    g = data.groupby(['location', 'district', 'latitude', 'longitude']).agg(
        ads=('price', 'size'), median_price=('price', 'median'), per_sqft=('price_per_sqft', 'median')).reset_index()
    g['median_m'] = (g['median_price'] / 1e6).round(1)
    g['per_sqft'] = g['per_sqft'].round(0)
    return g


def colour_scale(values):
    # light green to deep green, scaled between the 5th and 95th percentile
    lo, hi = np.percentile(values, [5, 95])
    t = np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)
    light, dark = np.array([207, 229, 223]), np.array([15, 92, 80])
    return [[int(c) for c in light + (dark - light) * x] + [210] for x in t]


TOOLTIP = {
    'html': '<b>{location}</b>, {district}<br/>Median Rs {median_m} M<br/>Rs {per_sqft} per sqft<br/>{ads} ads',
    'style': {'backgroundColor': '#0F3D36', 'color': 'white', 'fontFamily': 'Manrope', 'fontSize': '13px'},
}

# ---------- state ----------
# one list of areas the model knows, busiest first
known = set((d, l) for d in meta['locations'] for l in meta['locations'][d])
ad_counts = houses.groupby(['district', 'location']).size()
areas = sorted(known, key=lambda a: -ad_counts.get(a, 0))
# the ads cover mostly oct to mar, other months are too thin for the model
MONTH_OPTIONS = [MONTHS[m - 1] for _, m in TIMELINE]
today = MONTHS[date.today().month - 1]
DEFAULTS = {
    'area': areas[0],
    'bedrooms': 3, 'bathrooms': 2, 'house_size_sqft': 2000, 'land_size_perches': 10.0,
    'membership_level': 'free', 'is_member': 'No', 'is_verified': 'No',
    'month': today if today in MONTH_OPTIONS else MONTH_OPTIONS[0],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_sample():
    f = sample['features']
    st.session_state.update({
        'area': (f['district'], f['location']),
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


def area_label(area):
    district, location = area
    return f'{location}, {district} ({ad_counts.get(area, 0):,} ads)'


def typical(col):
    lo, hi = houses[col].quantile([.05, .95])
    return f'Most ads have {lo:,.0f} to {hi:,.0f}.'


def map_clicked():
    # runs before the rerun, so it is allowed to change the widget values
    picked = st.session_state['estimate_map'].selection.objects.get('places', [])
    if picked:
        st.session_state['area'] = (picked[0]['district'], picked[0]['location'])
        reset_flag()


# ---------- header ----------
st.markdown("""
<div class="hero">
  <div class="brand"><div class="logo">P</div><div class="name">Perch</div></div>
  <h1>What is a house worth, anywhere in Sri Lanka?</h1>
  <p>Pick an area, describe the house and get an asking price estimate. Or explore how prices differ from
  town to town and how they moved over the months. Built from thousands of ikman.lk house ads.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="body">', unsafe_allow_html=True)
tab_est, tab_map, tab_about = st.tabs(['Estimate a price', 'Price map', 'About the model'])

# =====================================================================
# tab 1: estimate
# =====================================================================
with tab_est:
    left, right = st.columns([3, 2], gap='large')

    with left:
        with st.container(key='card_map'):
            st.markdown('### 1. Choose the area')
            st.caption('Click a dot to pick the area. Bigger dots have more ads.')

            places = area_stats(houses)
            places = places[[(d, l) in known for d, l in zip(places['district'], places['location'])]].copy()
            places['radius'] = 900 + np.sqrt(places['ads']) * 160
            places['colour'] = colour_scale(places['median_price'].values)

            picked_district, picked_location = st.session_state['area']
            current = places[(places['district'] == picked_district) & (places['location'] == picked_location)]
            layers = [pdk.Layer('ScatterplotLayer', data=places, id='places', pickable=True,
                                get_position='[longitude, latitude]', get_radius='radius', radius_min_pixels=5,
                                radius_max_pixels=28, get_fill_color='colour', stroked=True,
                                get_line_color=[255, 255, 255], line_width_min_pixels=1)]
            if len(current):
                layers.append(pdk.Layer('ScatterplotLayer', data=current, get_position='[longitude, latitude]',
                                        get_radius='radius', radius_min_pixels=9, radius_max_pixels=34,
                                        get_fill_color=[242, 184, 75, 230], stroked=True,
                                        get_line_color=[16, 35, 31], line_width_min_pixels=2))
                view = pdk.ViewState(latitude=float(current['latitude'].iloc[0]),
                                     longitude=float(current['longitude'].iloc[0]), zoom=9.5)
            else:
                view = pdk.ViewState(latitude=7.6, longitude=80.7, zoom=6.6)

            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style='light', tooltip=TOOLTIP),
                            on_select=map_clicked, selection_mode='single-object', key='estimate_map', height=380,
                            alt='Map of Sri Lanka with a dot for every area that has house ads. Bigger dots have more ads and '
                                'darker dots have higher median prices. The Area list below makes the same choice.')

        st.write('')
        with st.container(key='card_form'):
            st.markdown('### 2. Describe the house')
            district, location = st.selectbox('Area', areas, key='area', format_func=area_label, on_change=reset_flag,
                                              help='Type to search. Clicking a dot on the map sets this too.')

            c3, c4 = st.columns(2)
            bedrooms = c3.slider('Bedrooms', 1, 8, key='bedrooms', on_change=reset_flag, help=typical('bedrooms'))
            bathrooms = c4.slider('Bathrooms', 1, 6, key='bathrooms', on_change=reset_flag, help=typical('bathrooms'))
            house_size = st.slider('House size (sqft)', 300, 8000, step=50, key='house_size_sqft',
                                   on_change=reset_flag, help=typical('house_size_sqft'))
            land_size = st.slider('Land size (perches)', 1.0, 100.0, step=0.5, key='land_size_perches',
                                  on_change=reset_flag, help=typical('land_size_perches'))

            # soft checks, the estimate still shows
            notes = []
            if house_size / bedrooms < 200:
                notes.append(f'{bedrooms} bedrooms in {house_size:,} sqft is very tight, check the numbers.')
            if bathrooms > bedrooms + 1:
                notes.append('More bathrooms than bedrooms is unusual.')
            if house_size > houses['house_size_sqft'].quantile(.99):
                notes.append('Houses this big are rare in the ads, so the estimate is rough.')
            if land_size > houses['land_size_perches'].quantile(.99):
                notes.append('Plots this big are rare in the ads, so the estimate is rough.')
            if ad_counts.get((district, location), 0) < 100:
                notes.append(f'Only {ad_counts.get((district, location), 0)} ads in {location}, so the estimate is rough.')
            for note in notes:
                st.warning(note, icon=':material/info:')

            with st.expander('Listing details'):
                st.caption('Who posted the ad and when. These move the asking price a little, '
                           'so leave them as they are unless you know them.')
                c7, c8, c9, c10 = st.columns(4)
                membership = c7.selectbox('Membership', meta['membership_levels'], key='membership_level', on_change=reset_flag)
                is_member = c8.selectbox('Member', ['No', 'Yes'], key='is_member', on_change=reset_flag)
                is_verified = c9.selectbox('Verified', ['No', 'Yes'], key='is_verified', on_change=reset_flag)
                month = c10.selectbox('Posted in', MONTH_OPTIONS, key='month', on_change=reset_flag)

            st.button('Try a real listing the model never saw', on_click=load_sample)

        extras = st.container()

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

    # the price and the pin slide from the last estimate, they only move when it changed
    quantiles = meta['price_quantiles'].get(district)
    pct = float(np.interp(price, quantiles, np.arange(101))) if quantiles else None
    tenths = round(price / 1e5)
    pos = min(max(pct, 2), 98) if quantiles else 50
    last_tenths, last_pos = st.session_state.get('shown', (round(tenths * .8), pos))
    st.session_state['shown'] = (tenths, pos)
    moved = ' reveal' if tenths != last_tenths else ''

    with right:
        with st.container(key='result_sticky'):
            st.markdown(f"""
            <div class="card est{moved}" id="result" role="status">
              <div class="label">Estimated asking price</div>
              <div class="big count" role="img" aria-label="{short(price)}" style="--to: {tenths}; --from: {last_tenths}"></div>
              <div class="sub">Rs {price:,.0f}</div>
              <div class="range">Likely range: {short(low)} to {short(high)}<br>
              <span class="note">About 2 in 3 ads like this one fall inside it.</span></div>
            </div>
            """, unsafe_allow_html=True)

            if quantiles:
                st.markdown(f"""
                <div class="card" style="margin-top: 1rem">
                  <div class="label">Against {district} listings</div>
                  <div class="bar"><div class="pin" style="--pos: {pos}%; --pin-from: {last_pos}%"></div></div>
                  <div class="bar-labels"><span>{short(quantiles[2])}</span><span>median {short(quantiles[50])}</span><span>{short(quantiles[98])}</span></div>
                  <div class="sub" style="margin-top: .6rem">Priced above about {pct:.0f}% of house ads in {district}.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="tiles" style="margin-top: 1rem">
              <div class="tile"><div class="label">Per sqft</div><div class="v">Rs {price / house_size:,.0f}</div></div>
              <div class="tile"><div class="label">Per perch of land</div><div class="v">{short(price / land_size)}</div></div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.get('sample_loaded'):
                actual = sample['actual_price']
                err = abs(price - actual) / actual * 100
                st.markdown(f"""
                <div class="check"><b>A real listing the model never saw.</b>
                The real asking price was {short(actual)}, so the estimate is off by {err:.1f}%.</div>
                """, unsafe_allow_html=True)

    with extras:
        # the same house in other busy areas
        busy = (houses.groupby(['location', 'district']).size().rename('ads').reset_index()
                .query('ads >= 100').sort_values('ads', ascending=False).head(12))
        busy = busy[[(d, l) in known for d, l in zip(busy['district'], busy['location'])]]
        what_if = pd.DataFrame([make_row(location=l, district=d) for l, d in zip(busy['location'], busy['district'])])
        compare = pd.Series(model.predict(what_if) / 1e6, index=busy['location'].values, name='Rs million')

        with st.expander('Compare with other areas and sizes'):
            st.subheader('The same house elsewhere')
            st.bar_chart(compare.sort_values(ascending=False), horizontal=True, sort=False, color=GREEN, height=330,
                         alt='Bar chart of the estimated price in Rs million for this house in the twelve busiest areas.')

            # price against house size for this location
            sizes = np.arange(500, 6001, 250)
            curve = pd.DataFrame([make_row(house_size_sqft=int(s)) for s in sizes])
            st.subheader('Price as the house gets bigger')
            st.line_chart(pd.Series(model.predict(curve) / 1e6, index=sizes, name='Rs million'), color=SAFFRON, height=240,
                          alt='Line chart of the estimated price in Rs million as house size grows from 500 to 6,000 sqft.')

        st.caption(f'Estimates are based on asking prices from ads, not final sale prices. Model: {meta["model"]}.')

    # on a phone the card is far below the form, so the price stays pinned at the bottom
    st.markdown(f'<div class="price-bar"><span>Estimated asking price</span><b>{short(price)}</b></div>', unsafe_allow_html=True)

# =====================================================================
# tab 2: price map with timeline
# =====================================================================
with tab_map:
    with st.container(key='card_controls'):
        st.markdown('### Explore prices across the island')
        a, b, c = st.columns([2, 2, 2])
        metric = a.selectbox('Show', ['Median price', 'Price per sqft', 'Number of ads'], key='metric')
        style = b.selectbox('Map style', ['3D columns', 'Heatmap', 'Bubbles'], key='style')
        beds = c.select_slider('Bedrooms', options=[1, 2, 3, 4, 5, 6], value=(1, 6), key='beds')
        d, e = st.columns([5, 1])
        month_label = d.select_slider('Timeline', options=['All months'] + TIMELINE_LABELS, key='timeline')
        e.write('')
        play = e.button('Play')

    data = houses[houses['bedrooms'].between(*beds)]
    value_col = {'Median price': 'median_price', 'Price per sqft': 'per_sqft', 'Number of ads': 'ads'}[metric]

    def draw(label, slot):
        if label == 'All months':
            part = data
        else:
            y, m = TIMELINE[TIMELINE_LABELS.index(label)]
            part = data[data['posted_month'] == m]
        stats = area_stats(part)
        stats = stats[stats['ads'] >= (3 if label != 'All months' else 5)].copy()
        if stats.empty:
            slot.info('No areas have enough ads for this filter.')
            return stats
        stats['colour'] = colour_scale(stats[value_col].values)
        stats['elevation'] = stats[value_col] / stats[value_col].max() * 90000
        stats['radius'] = 1000 + np.sqrt(stats['ads']) * 180
        stats['value'] = stats[value_col]

        if style == '3D columns':
            layer = pdk.Layer('ColumnLayer', data=stats, get_position='[longitude, latitude]', get_elevation='elevation',
                              radius=1800, get_fill_color='colour', pickable=True, extruded=True, auto_highlight=True)
            view = pdk.ViewState(latitude=7.2, longitude=80.2, zoom=8.2, pitch=55, bearing=-15)
        elif style == 'Heatmap':
            layer = pdk.Layer('HeatmapLayer', data=stats, get_position='[longitude, latitude]', get_weight='value',
                              aggregation='"MEAN"', radius_pixels=55, opacity=.85,
                              color_range=[[207, 229, 223], [140, 199, 180], [242, 184, 75], [226, 120, 50], [158, 40, 40], [90, 10, 30]])
            view = pdk.ViewState(latitude=7.4, longitude=80.5, zoom=7.3)
        else:
            layer = pdk.Layer('ScatterplotLayer', data=stats, get_position='[longitude, latitude]', get_radius='radius',
                              radius_min_pixels=5, radius_max_pixels=40, get_fill_color='colour', pickable=True,
                              stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)
            view = pdk.ViewState(latitude=7.4, longitude=80.5, zoom=7.3)

        slot.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='light', tooltip=TOOLTIP), height=520,
                          alt=f'Map of Sri Lanka showing {metric.lower()} by area as {style.lower()}, for {label.lower()}. The table below lists the same areas.')
        return stats

    st.write('')
    map_slot = st.empty()
    if play:
        for label in TIMELINE_LABELS:
            map_slot.caption(f'Showing {label}')
            shown = draw(label, st.empty())
            time.sleep(1.2)
        shown = draw(TIMELINE_LABELS[-1], map_slot)
    else:
        shown = draw(month_label, map_slot)

    if len(shown):
        solid = shown[shown['ads'] >= 15]
        top = solid.sort_values('median_price', ascending=False).iloc[0] if len(solid) else None
        low_row = solid.sort_values('median_price').iloc[0] if len(solid) else None
        st.markdown(f"""
        <div class="kpis" style="margin-top: 1rem">
          <div class="kpi"><div class="label">Areas shown</div><div class="v">{len(shown)}</div><div class="s">{int(shown['ads'].sum()):,} ads</div></div>
          <div class="kpi"><div class="label">Median price</div><div class="v">{short(float(np.average(shown['median_price'], weights=shown['ads'])))}</div><div class="s">weighted by ads</div></div>
          <div class="kpi"><div class="label">Priciest area</div><div class="v">{top['location'] if top is not None else '-'}</div><div class="s">{short(top['median_price']) if top is not None else ''}</div></div>
          <div class="kpi"><div class="label">Cheapest area</div><div class="v">{low_row['location'] if low_row is not None else '-'}</div><div class="s">{short(low_row['median_price']) if low_row is not None else ''}</div></div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Median price by month')
        pick = st.multiselect('Compare districts', sorted(houses['district'].unique()), default=['Colombo', 'Gampaha', 'Kandy'], key='trend_districts')
        lines = {}
        for name, sub in [('All island', data)] + [(p, data[data['district'] == p]) for p in pick]:
            by_month = sub.groupby('posted_month')['price'].agg(['median', 'size'])
            by_month = by_month[by_month['size'] >= 30]
            lines[name] = pd.Series([by_month['median'].get(m, np.nan) / 1e6 for _, m in TIMELINE], index=TIMELINE_LABELS)
        st.line_chart(pd.DataFrame(lines), height=300,
                      alt='Line chart of the median asking price in Rs million for each month, for the whole island and the chosen districts.')
        st.caption('Median asking price in Rs million. A month is left out for a district when it has fewer than 30 ads.')
    with c2:
        st.subheader('Most and least expensive areas')
        ranked = area_stats(data)
        ranked = ranked[ranked['ads'] >= 15].sort_values('median_price', ascending=False)
        table = ranked[['location', 'district', 'ads', 'median_m', 'per_sqft']].rename(
            columns={'location': 'Area', 'district': 'District', 'ads': 'Ads', 'median_m': 'Median (Rs M)', 'per_sqft': 'Rs per sqft'})
        st.write('Top 8')
        st.dataframe(table.head(8), hide_index=True, width='stretch')
        st.write('Bottom 8')
        st.dataframe(table.tail(8), hide_index=True, width='stretch')

# =====================================================================
# tab 3: about
# =====================================================================
with tab_about:
    st.subheader('How the estimate is made')
    st.markdown(f"""
    The price comes from a pickled scikit-learn pipeline: preprocessing (imputing, scaling, one hot encoding of
    locations) followed by **{meta['model']}**, trained on house ads from ikman.lk. The model predicts the log of the
    price, and the range shown is one test error either side (about 68% of ads fall inside it).

    - Reposted ads were removed before training so the test score is not flattering.
    - Prices are asking prices, not what the house finally sold for.
    - Ads cover Sep 2022 to Apr 2023 only, so the timeline is short and says nothing about longer trends.
    """)

    results = pd.DataFrame(meta['results']).set_index('model')
    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Model comparison')
        st.dataframe(results.round(3), width='stretch')
        st.caption('rmsle is the typical miss on the log price, lower is better. r2 is how much of the price spread the model explains, higher is better. mae is the average miss in Rs million. All scored on 20% of the ads kept aside.')
    with c2:
        st.subheader('What drives the price')
        importance = pd.DataFrame(meta['importance']).sort_values('xgboost (tuned)', ascending=False)
        st.bar_chart(importance, horizontal=True, sort=False, stack=False, color=[GREEN, SAFFRON], height=330,
                     alt='Bar chart of how much each feature matters to the random forest and to xgboost.')
        st.caption('How much the error grows when a feature is shuffled, so longer means more important.')

st.markdown('</div>', unsafe_allow_html=True)
