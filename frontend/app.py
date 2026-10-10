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


model = load_model()
meta = load_json('model_meta.json')
sample = load_json('holdout_sample.json')
houses = load_houses()


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


def map_clicked():
    # runs before the rerun, so it is allowed to change the widget values
    picked = st.session_state['estimate_map'].selection.objects.get('places', [])
    if picked:
        st.session_state['district'] = picked[0]['district']
        st.session_state['location'] = picked[0]['location']
        reset_flag()


# ---------- header ----------
st.title('Perch')
st.write('Describe a house and get a price estimate.')

tab_est, tab_map = st.tabs(['Estimate a price', 'Price map'])

with tab_est:
    known = set((d, l) for d in meta['locations'] for l in meta['locations'][d])
    left, right = st.columns([3, 2], gap='large')

    with left:
        st.subheader('Choose the location')
        st.caption('Click a dot to pick the area. Bigger dots have more ads.')

        places = area_stats(houses)
        places = places[[(d, l) in known for d, l in zip(places['district'], places['location'])]].copy()
        places['radius'] = 900 + np.sqrt(places['ads']) * 160
        places['colour'] = colour_scale(places['median_price'].values)

        current = places[places['location'] == st.session_state['location']]
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
                        on_select=map_clicked, selection_mode='single-object', key='estimate_map', height=380)

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

with tab_map:
    with st.container():
        st.subheader('Explore prices across the island')
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

        slot.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='light', tooltip=TOOLTIP), height=520)
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
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Areas shown', len(shown), f"{int(shown['ads'].sum()):,} ads", delta_color='off')
        k2.metric('Median price', short(float(np.average(shown['median_price'], weights=shown['ads']))), 'weighted by ads', delta_color='off')
        k3.metric('Priciest area', top['location'] if top is not None else '-', short(top['median_price']) if top is not None else '', delta_color='off')
        k4.metric('Cheapest area', low_row['location'] if low_row is not None else '-', short(low_row['median_price']) if low_row is not None else '', delta_color='off')

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Median price by month')
        pick = st.multiselect('Compare districts', sorted(houses['district'].unique()), default=['Colombo', 'Gampaha', 'Kandy'], key='trend_districts')
        lines = {}
        for name, sub in [('All island', data)] + [(p, data[data['district'] == p]) for p in pick]:
            by_month = sub.groupby('posted_month')['price'].agg(['median', 'size'])
            by_month = by_month[by_month['size'] >= 30]
            lines[name] = pd.Series([by_month['median'].get(m, np.nan) / 1e6 for _, m in TIMELINE], index=TIMELINE_LABELS)
        st.line_chart(pd.DataFrame(lines), height=300)
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
