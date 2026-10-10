import pickle
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from shared import (ROOT, TIMELINE, TIMELINE_LABELS, GREEN, SAFFRON, TOOLTIP_STYLE,
                    colour_scale, inject_css, load_json, short, topbar)

inject_css()
topbar()

TREES = ['rf', 'xgb', 'lgbm', 'cat']
MODELS = {
    'Stacking': ('stacking', 'Learns how much to trust each model, with ridge and a map based model added. Lowest error in the tests.'),
    'XGBoost': ('xgboost', 'Boosted trees that fix each other\'s mistakes. Best single model.'),
    'LightGBM': ('lightgbm', 'Fast boosted trees that read location as a category.'),
    'CatBoost': ('catboost', 'Boosted trees with built in handling of categories.'),
    'Random forest': ('random forest', 'Many deep trees averaged.'),
    'Voting': ('voting', 'Plain average of the four tree models.'),
}
BASE_NAME = {'xgboost': 'xgb', 'lightgbm': 'lgbm', 'catboost': 'cat', 'random forest': 'rf'}
WORDS = {'main_road': 'Main road', 'highway': 'Near highway', 'clear_deed': 'Clear deed', 'near_town': 'Near town',
         'bare_land': 'Bare land', 'water': 'Water', 'electricity': 'Electricity', 'lake_or_beach': 'Lake or beach',
         'paddy': 'Paddy', 'investment': 'Investment'}
USES = ['Residential', 'Commercial', 'Agricultural']


# ---------- loading ----------
@st.cache_resource
def load_model():
    with open(ROOT / 'model' / 'land_model.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_land():
    land = pd.read_csv(ROOT / 'data' / 'processed' / 'land_clean.csv')
    coords = pd.read_csv(ROOT / 'data' / 'processed' / 'location_coords.csv')
    return land.merge(coords[['location', 'latitude', 'longitude']], on='location', how='left')


try:
    meta = load_json('land_meta.json')
    sample = load_json('land_holdout_sample.json')
    land = load_land()
    model = load_model()
except FileNotFoundError:
    st.error('The land model or data files are missing. Run the land notebooks first, they save everything this page needs.')
    st.stop()


def predict(frame, choice):
    # the saved stack holds every base model, they all predict log price
    if choice == 'stacking':
        return model.predict(frame)
    logs = {n: est.predict(frame) for n, est in model.regressor_.named_estimators_.items()}
    if choice == 'voting':
        return np.exp(np.mean([logs[n] for n in TREES], axis=0))
    return np.exp(logs[BASE_NAME[choice]])


def rs(value):
    return f'Rs {value:,.0f}'


def area_stats(data):
    g = data.groupby(['location', 'district', 'latitude', 'longitude']).agg(
        ads=('price_per_perch', 'size'), median_price=('price_per_perch', 'median'),
        median_size=('land_size_perches', 'median')).reset_index()
    g['median_total'] = g['median_price'] * g['median_size']
    g['median_k'] = (g['median_price'] / 1e3).round(0).astype(int)
    return g


TOOLTIP = {
    'html': '<b>{location}</b>, {district}<br/>Median Rs {median_k} k per perch<br/>{ads} ads',
    'style': TOOLTIP_STYLE,
}

# ---------- state ----------
known = set((d, l) for d in meta['locations'] for l in meta['locations'][d])
ad_counts = land.groupby(['district', 'location']).size()
areas = sorted(known, key=lambda a: -ad_counts.get(a, 0))
coord_of = land.drop_duplicates('location').set_index('location')[['latitude', 'longitude']]
DEFAULTS = {'land_area': areas[0], 'land_size': 10.0, 'land_use': ['Residential'], 'land_words': [],
            'land_membership': 'free', 'land_member': 'No', 'land_verified': 'No'}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_sample():
    f = sample['features']
    st.session_state.update({
        'land_area': (f['district'], f['location']), 'land_size': float(f['land_size_perches']),
        'land_use': [u for u in USES if f[u.lower()]],
        'land_words': [WORDS[k] for k in WORDS if f[k]],
        'land_membership': f['membership_level'],
        'land_member': 'Yes' if f['is_member'] else 'No',
        'land_verified': 'Yes' if f['is_verified'] else 'No',
        'land_sample': True,
    })


def reset_flag():
    st.session_state['land_sample'] = False


def area_label(area):
    district, location = area
    return f'{location}, {district} ({ad_counts.get(area, 0):,} ads)'


def map_clicked():
    picked = st.session_state['land_map'].selection.objects.get('places', [])
    if picked:
        st.session_state['land_area'] = (picked[0]['district'], picked[0]['location'])
        reset_flag()


# ---------- header ----------
st.markdown("""
<div class="hero">
  <h1>What is a plot of land worth, per perch?</h1>
  <p>Pick an area, say how big the plot is and what the ad says about it. Perch gives a price per perch
  and a total. Or explore how land prices differ from town to town. Built from ikman.lk land ads.</p>
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

            places = area_stats(land)
            places = places[[(d, l) in known for d, l in zip(places['district'], places['location'])]].copy()
            places['radius'] = 900 + np.sqrt(places['ads']) * 120
            places['colour'] = colour_scale(places['median_price'].values)

            picked_district, picked_location = st.session_state['land_area']
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
                            on_select=map_clicked, selection_mode='single-object', key='land_map', height=380,
                            alt='Map of Sri Lanka with a dot for every area that has land ads. Bigger dots have more ads and '
                                'darker dots have higher median prices per perch. The Area list below makes the same choice.')

        st.write('')
        with st.container(key='card_form'):
            st.markdown('### 2. Describe the plot')
            district, location = st.selectbox('Area', areas, key='land_area', format_func=area_label, on_change=reset_flag,
                                              help='Type to search. Clicking a dot on the map sets this too.')
            lo, hi = land['land_size_perches'].quantile([.05, .95])
            size = st.slider('Land size (perches)', 1.0, 100.0, step=0.5, key='land_size', on_change=reset_flag,
                             help=f'Most ads have {lo:,.0f} to {hi:,.0f} perches. 1 acre is 160 perches.')
            use = st.pills('Land use', USES, selection_mode='multi', key='land_use', on_change=reset_flag)
            words = st.pills('What the ad says', list(WORDS.values()), selection_mode='multi', key='land_words',
                             on_change=reset_flag, help='Words sellers use. Pick the ones that describe this plot.')

            notes = []
            if size > meta['size_range'][1]:
                notes.append('Plots this big are rare in the ads, so the estimate is rough.')
            if ad_counts.get((district, location), 0) < 100:
                notes.append(f'Only {ad_counts.get((district, location), 0)} ads in {location}, so the estimate is rough.')
            if not use:
                notes.append('No land use picked, the estimate treats it as other land.')
            for note in notes:
                st.warning(note, icon=':material/info:')

            with st.expander('Seller details'):
                st.caption('Who posted the ad. These move the asking price a little, so leave them as they are unless you know them.')
                c1, c2, c3 = st.columns(3)
                membership = c1.selectbox('Membership', meta['membership_levels'], key='land_membership', on_change=reset_flag)
                is_member = c2.selectbox('Member', ['No', 'Yes'], key='land_member', on_change=reset_flag)
                is_verified = c3.selectbox('Verified', ['No', 'Yes'], key='land_verified', on_change=reset_flag)

            label = st.segmented_control('Model', list(MODELS), default='Stacking', key='land_model_choice',
                                         help='All six come from the same training data and are scored on the same test set.')
            label = label or 'Stacking'
            model_name, model_note = MODELS[label]
            st.caption(f'{model_note} Test error {meta["test_rmsle"][model_name]:.3f} rmsle.')

            st.button('Try a real listing the model never saw', on_click=load_sample, key='land_try')

        extras = st.container()

    def make_row(**changes):
        lat, lon = coord_of.loc[location, ['latitude', 'longitude']]
        row = {
            'land_size_perches': size, 'latitude': lat, 'longitude': lon,
            'is_verified': int(is_verified == 'Yes'), 'is_member': float(is_member == 'Yes'),
            'residential': int('Residential' in use), 'commercial': int('Commercial' in use),
            'agricultural': int('Agricultural' in use),
            'location': location, 'district': district, 'membership_level': membership,
        }
        row.update({k: int(WORDS[k] in words) for k in WORDS})
        row.update(changes)
        return row

    price = float(predict(pd.DataFrame([make_row()]), model_name)[0])
    spread = np.exp(meta['test_rmsle'][model_name])
    low, high = price / spread, price * spread

    quantiles = meta['price_quantiles'].get(district)
    pct = float(np.interp(price, quantiles, np.arange(101))) if quantiles else None
    pos = min(max(pct, 2), 98) if quantiles else 50
    last_pos = st.session_state.get('land_shown', pos)
    st.session_state['land_shown'] = pos

    with right:
        with st.container(key='result_sticky'):
            st.markdown(f"""
            <div class="card est" id="result" role="status">
              <div class="label">Estimated price per perch</div>
              <div class="big">{rs(price)}</div>
              <div class="sub">{size:g} perches comes to {short(price * size)}</div>
              <div class="range">Likely range: {rs(low)} to {rs(high)} per perch<br>
              <span class="note">About 2 in 3 ads like this one fall inside it.</span></div>
            </div>
            """, unsafe_allow_html=True)

            if quantiles:
                st.markdown(f"""
                <div class="card" style="margin-top: 1rem">
                  <div class="label">Against {district} listings</div>
                  <div class="bar"><div class="pin" style="--pos: {pos}%; --pin-from: {last_pos}%"></div></div>
                  <div class="bar-labels"><span>{rs(quantiles[2])}</span><span>median {rs(quantiles[50])}</span><span>{rs(quantiles[98])}</span></div>
                  <div class="sub" style="margin-top: .6rem">Priced above about {pct:.0f}% of land ads in {district}.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="tiles" style="margin-top: 1rem">
              <div class="tile"><div class="label">Total price</div><div class="v">{short(price * size)}</div></div>
              <div class="tile"><div class="label">Per acre</div><div class="v">{short(price * 160)}</div></div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.get('land_sample'):
                actual = sample['actual_price_per_perch']
                err = abs(price - actual) / actual * 100
                st.markdown(f"""
                <div class="check"><b>A real listing the model never saw.</b>
                The real asking price was {rs(actual)} per perch, so the estimate is off by {err:.1f}%.</div>
                """, unsafe_allow_html=True)

    with extras:
        busy = (land.groupby(['location', 'district']).size().rename('ads').reset_index()
                .query('ads >= 200').sort_values('ads', ascending=False).head(12))
        busy = busy[[(d, l) in known for d, l in zip(busy['district'], busy['location'])]]
        what_if = pd.DataFrame([make_row(location=l, district=d, latitude=coord_of.loc[l, 'latitude'], longitude=coord_of.loc[l, 'longitude'])
                                for l, d in zip(busy['location'], busy['district'])])
        compare = pd.Series(predict(what_if, model_name) / 1e3, index=busy['location'].values, name='Rs thousand per perch')

        with st.expander('Compare with other areas and sizes'):
            st.subheader('The same plot elsewhere')
            st.bar_chart(compare.sort_values(ascending=False), horizontal=True, sort=False, color=GREEN, height=330,
                         alt='Bar chart of the estimated price per perch in Rs thousand for this plot in the busiest areas.')

            sizes = np.array([2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100], dtype=float)
            curve = pd.DataFrame([make_row(land_size_perches=s) for s in sizes])
            st.subheader('Price per perch as the plot gets bigger')
            st.line_chart(pd.Series(predict(curve, model_name) / 1e3, index=sizes, name='Rs thousand per perch'), color=SAFFRON, height=240,
                          alt='Line chart of the estimated price per perch in Rs thousand as the plot grows from 2 to 100 perches.')

        st.caption(f'Estimates are based on asking prices from ads, not final sale prices. Model: {label}.')

# =====================================================================
# tab 2: price map with timeline
# =====================================================================
with tab_map:
    with st.container(key='card_controls'):
        st.markdown('### Explore land prices across the island')
        a, b, c, d2 = st.columns(4)
        metric = a.selectbox('Show', ['Median price per perch', 'Median total price', 'Number of ads'], key='land_metric')
        style = b.selectbox('Map style', ['3D columns', 'Heatmap', 'Bubbles'], key='land_style')
        use_filter = c.selectbox('Land use', ['All', 'Residential', 'Commercial', 'Agricultural'], key='land_use_filter')
        sizes_range = d2.select_slider('Plot size (perches)', options=[1, 5, 10, 15, 25, 50, 100, 200], value=(1, 200), key='land_sizes')
        d, e = st.columns([5, 1])
        month_label = d.select_slider('Timeline', options=['All months'] + TIMELINE_LABELS, key='land_timeline')
        e.write('')
        play = e.button('Play', key='land_play')

    data = land[land['land_size_perches'].between(*sizes_range)]
    if use_filter != 'All':
        data = data[data[use_filter.lower()] == 1]
    value_col = {'Median price per perch': 'median_price', 'Median total price': 'median_total', 'Number of ads': 'ads'}[metric]

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
        stats['radius'] = 1000 + np.sqrt(stats['ads']) * 140
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
          <div class="kpi"><div class="label">Median per perch</div><div class="v">{rs(float(np.average(shown['median_price'], weights=shown['ads'])))}</div><div class="s">weighted by ads</div></div>
          <div class="kpi"><div class="label">Priciest area</div><div class="v">{top['location'] if top is not None else '-'}</div><div class="s">{rs(top['median_price']) if top is not None else ''}</div></div>
          <div class="kpi"><div class="label">Cheapest area</div><div class="v">{low_row['location'] if low_row is not None else '-'}</div><div class="s">{rs(low_row['median_price']) if low_row is not None else ''}</div></div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Median price per perch by month')
        pick = st.multiselect('Compare districts', sorted(land['district'].unique()), default=['Colombo', 'Gampaha', 'Kalutara'], key='land_trend')
        lines = {}
        for name, sub in [('All island', data)] + [(p, data[data['district'] == p]) for p in pick]:
            by_month = sub.groupby('posted_month')['price_per_perch'].agg(['median', 'size'])
            by_month = by_month[by_month['size'] >= 30]
            lines[name] = pd.Series([by_month['median'].get(m, np.nan) / 1e3 for _, m in TIMELINE], index=TIMELINE_LABELS)
        st.line_chart(pd.DataFrame(lines), height=300,
                      alt='Line chart of the median asking price per perch in Rs thousand for each month, for the whole island and the chosen districts.')
        st.caption('Median asking price per perch in Rs thousand. A month is left out for a district when it has fewer than 30 ads.')
    with c2:
        st.subheader('Most and least expensive areas')
        ranked = area_stats(data)
        ranked = ranked[ranked['ads'] >= 15].sort_values('median_price', ascending=False)
        table = ranked[['location', 'district', 'ads', 'median_k', 'median_size']].rename(
            columns={'location': 'Area', 'district': 'District', 'ads': 'Ads', 'median_k': 'Median (Rs k per perch)', 'median_size': 'Median size'})
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
    The price per perch comes from a stack of tree models (random forest, xgboost, lightgbm and catboost) plus
    a ridge regression and a nearest neighbour model on the map, trained on land ads from ikman.lk. The stack
    learns how much to trust each one. You can also pick a single model on the first tab. The models predict the log of the price per perch,
    and the range shown is one test error either side (about 68% of ads fall inside it).

    - About 70% of land ads were reposts, only one of each was kept so the test score is not flattering.
    - Prices are asking prices, not what the plot finally sold for.
    - Ads cover Sep 2022 to Apr 2023 only. Land is much harder to predict than houses: the typical miss is about 32%.
    """)

    results = pd.DataFrame(meta['results']).set_index('model').sort_values('rmsle')
    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('Model comparison')
        st.dataframe(results.round(3), width='stretch')
        st.caption('rmsle is the typical miss on the log price, lower is better. r2 is how much of the price spread the model explains. mae is in Rs thousand per perch. Scored on 20% of the ads kept aside. District median, location median, ridge and knn are the simple approaches to beat.')
    with c2:
        st.subheader('What drives the price')
        importance = pd.Series(meta['importance']).sort_values(ascending=False)
        st.bar_chart(importance, horizontal=True, sort=False, color=GREEN, height=420,
                     alt='Bar chart of the average effect of each feature on the predicted log price per perch.')
        st.caption('Average shap value for xgboost, longer means the feature moves the price more.')

    c3, c4 = st.columns(2, gap='large')
    with c3:
        st.subheader('Leaving features out')
        st.dataframe(pd.DataFrame(meta['ablation']).set_index('what is left out').round(3), width='stretch')
        st.caption('xgboost with one group of features removed. Location and coordinates carry the same information, so removing one costs little but removing both costs a lot.')
    with c4:
        st.subheader('Weights inside the stack')
        weights = pd.Series(meta['stack_weights'], name='weight').round(3)
        st.dataframe(weights.to_frame(), width='stretch')
        st.caption('rf, xgb, lgbm and cat are the four tree models. Ridge gets no weight, the map based knn a small one.')

st.markdown('</div>', unsafe_allow_html=True)
