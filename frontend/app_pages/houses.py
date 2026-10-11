import pickle
import time
from datetime import date

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from shared import (ROOT, MONTHS, TIMELINE, TIMELINE_LABELS, GREEN, SAFFRON, TOOLTIP_STYLE,
                    colour_scale, hero, inject_css, load_json, model_bars, model_card, month_chart, short)

inject_css()

# ---------- loading ----------
@st.cache_resource
def load_model(name):
    with open(ROOT / 'model' / f'houses_{name}.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_houses():
    houses = pd.read_csv(ROOT / 'data' / 'processed' / 'houses_clean.csv')
    coords = pd.read_csv(ROOT / 'data' / 'processed' / 'location_coords.csv')
    houses = houses.merge(coords[['location', 'latitude', 'longitude']], on='location', how='left')
    houses['price_per_sqft'] = houses['price'] / houses['house_size_sqft']
    return houses


try:
    meta = load_json('houses_meta.json')
    load_model('stacking')
    sample = load_json('houses_holdout_sample.json')
    houses = load_houses()
except FileNotFoundError:
    st.error('The model or data files are missing. Run the notebooks first, they save everything the app needs.')
    st.stop()


RESULT_ROW = {'stacking': 'stacking (rf + xgb)', 'xgboost': 'xgboost (tuned)', 'random_forest': 'random forest (tuned)', 'voting': 'voting (rf + xgb)'}

MODELS = {
    'Stacking': ('stacking', 'Learns how much to trust the forest and xgboost. Lowest error in the tests.'),
    'XGBoost': ('xgboost', 'Boosted trees that fix each other\'s mistakes. Best single model.'),
    'Random forest': ('random_forest', 'Many deep trees averaged. Steady, slightly less accurate.'),
    'Voting': ('voting', 'Plain average of the random forest and xgboost.'),
}


def predict(frame, choice):
    # voting is the mean of the two log predictions, same as in the notebook
    if choice == 'voting':
        rf = load_model('random_forest').predict(frame)
        xgb = load_model('xgboost').predict(frame)
        return np.expm1((np.log1p(rf) + np.log1p(xgb)) / 2)
    return load_model(choice).predict(frame)


def area_stats(data):
    # one row per location with the numbers shown on the maps
    g = data.groupby(['location', 'district', 'latitude', 'longitude']).agg(
        ads=('price', 'size'), median_price=('price', 'median'), per_sqft=('price_per_sqft', 'median')).reset_index()
    g['median_m'] = (g['median_price'] / 1e6).round(1)
    g['per_sqft'] = g['per_sqft'].round(0)
    return g


TOOLTIP = {
    'html': '<b>{location}</b>, {district}<br/>Median Rs {median_m} M<br/>Rs {per_sqft} per sqft<br/>{ads} ads',
    'style': TOOLTIP_STYLE,
}

# ---------- state ----------
# one list of areas the model knows, busiest first
known = set((d, l) for d in meta['locations'] for l in meta['locations'][d])
ad_counts = houses.groupby(['district', 'location']).size()
areas = sorted(known, key=lambda a: -ad_counts.get(a, 0))
districts = sorted(meta['locations'])
# the ads cover mostly oct to mar, other months are too thin for the model
MONTH_OPTIONS = [MONTHS[m - 1] for _, m in TIMELINE]
today = MONTHS[date.today().month - 1]
DEFAULTS = {
    'district': areas[0][0], 'location': areas[0][1],
    'bedrooms': 3, 'bathrooms': 2, 'house_size_sqft': 2000, 'land_size_perches': 10.0,
    'membership_level': 'free', 'is_member': 'No', 'is_verified': 'No',
    'month': today if today in MONTH_OPTIONS else MONTH_OPTIONS[0],
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
    # a new district starts on its busiest city
    d = st.session_state['district']
    st.session_state['location'] = max(meta['locations'][d], key=lambda l: ad_counts.get((d, l), 0))
    reset_flag()


def typical(col):
    lo, hi = houses[col].quantile([.05, .95])
    return f'Most ads have {lo:,.0f} to {hi:,.0f}.'


def map_clicked():
    # runs before the rerun, so it is allowed to change the widget values
    picked = st.session_state['estimate_map'].selection.objects.get('places', [])
    if picked:
        st.session_state['district'] = picked[0]['district']
        st.session_state['location'] = picked[0]['location']
        reset_flag()


# ---------- header ----------
hero('What is a house worth, anywhere in Sri Lanka?',
     'Pick an area, describe the house and get an asking price estimate. Compare four models on the same house, '
     'or explore how prices differ from town to town and how they moved over the months.',
     [f'{len(houses):,} house ads', '4 models', 'Sep 2022 to Apr 2023'], current='houses')

tab_est, tab_compare, tab_map, tab_about = st.tabs(['Estimate a price', 'Compare models', 'Price map', 'About the model'])

# =====================================================================
# tab 1: estimate
# =====================================================================
with tab_est:
    left, right = st.columns([2, 3], gap='large')

    with left:
        with st.container(key='card_form'):
            st.markdown('### Describe the house')
            c1, c2 = st.columns(2)
            district = c1.selectbox('District', districts, key='district', on_change=district_changed)
            location = c2.selectbox('City', sorted(meta['locations'][district]), key='location', on_change=reset_flag,
                                    help='Type to search. Clicking a dot on the map sets both.')

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

            model_label = st.segmented_control('Model', list(MODELS), default='Stacking', key='model_choice',
                                               help='All four were trained on the same ads and scored on the same test set.')
            if model_label is None:
                model_label = 'Stacking'
            model_name, model_note = MODELS[model_label]
            st.caption(f'{model_note} Test error {meta["test_rmsle"][model_name.replace("_", " ")]:.3f} rmsle.')

            st.button('Try a real listing the model never saw', on_click=load_sample)

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

    price = float(predict(pd.DataFrame([make_row()]), model_name)[0])
    typical_miss = next(r['median abs % error'] for r in meta['results'] if r['model'] == RESULT_ROW[model_name])

    # about 68% of listings fall inside +/- one test rmsle on the log scale
    spread = np.exp(meta['test_rmsle'][model_name.replace('_', ' ')])
    low, high = price / spread, price * spread

    # the price and the pin slide from the last estimate, they only move when it changed
    quantiles = meta['price_quantiles'].get(district)
    pct = float(np.interp(price, quantiles, np.arange(101))) if quantiles else None
    tenths = round(price / 1e5)
    pos = min(max(pct, 2), 98) if quantiles else 50
    last_tenths, last_pos = st.session_state.get('shown', (round(tenths * .8), pos))
    st.session_state['shown'] = (tenths, pos)
    moved = ' reveal' if tenths != last_tenths else ''

    with left:
        if quantiles:
            st.markdown(f"""
            <div class="card" style="margin-top: .2rem">
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

    with right:
        st.markdown(f"""
        <div class="card est{moved}" id="result" role="status">
          <div class="label">Estimated asking price</div>
          <div class="big count" role="img" aria-label="{short(price)}" style="--to: {tenths}; --from: {last_tenths}"></div>
          <div class="sub">Rs {price:,.0f} <span class="by">by {model_label}</span></div>
          <div class="range">Likely range: {short(low)} to {short(high)}<br>
          <span class="note">About 2 in 3 ads like this one fall inside it.</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('sample_loaded'):
            actual = sample['actual_price']
            err = abs(price - actual) / actual * 100
            st.markdown(f"""
            <div class="check"><b>A real listing the model never saw.</b>
            The real asking price was {short(actual)}, so the estimate is off by {err:.1f}%.
            One listing is not a score: over the whole test set the typical miss of {model_label} is {typical_miss:.0f}%.</div>
            """, unsafe_allow_html=True)

        st.write('')
        with st.container(key='card_map'):
            st.markdown('### Pick on the map')
            st.caption('Click a dot to set the district and city. Bigger dots have more ads, darker dots cost more.')
            places = area_stats(houses)
            places = places[[(d, l) in known for d, l in zip(places['district'], places['location'])]].copy()
            places['radius'] = 900 + np.sqrt(places['ads']) * 160
            places['colour'] = colour_scale(places['median_price'].values)

            current = places[(places['district'] == district) & (places['location'] == location)]
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
                                     longitude=float(current['longitude'].iloc[0]), zoom=9)
            else:
                view = pdk.ViewState(latitude=7.6, longitude=80.7, zoom=6.6)

            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style='light', tooltip=TOOLTIP),
                            on_select=map_clicked, selection_mode='single-object', key='estimate_map', height=560,
                            alt='Map of Sri Lanka with a dot for every area that has house ads. Bigger dots have more ads and darker dots have higher median prices. The district and city lists make the same choice.')

    extras = st.container()
    with extras:
        # the same house in other busy areas
        busy = (houses.groupby(['location', 'district']).size().rename('ads').reset_index()
                .query('ads >= 100').sort_values('ads', ascending=False).head(12))
        busy = busy[[(d, l) in known for d, l in zip(busy['district'], busy['location'])]]
        what_if = pd.DataFrame([make_row(location=l, district=d) for l, d in zip(busy['location'], busy['district'])])
        compare = pd.Series(predict(what_if, model_name) / 1e6, index=busy['location'].values, name='Rs million')

        with st.expander('Compare with other areas and sizes'):
            st.subheader('The same house elsewhere')
            st.bar_chart(compare.sort_values(ascending=False), horizontal=True, sort=False, color=GREEN, height=330,
                         alt='Bar chart of the estimated price in Rs million for this house in the twelve busiest areas.')

            # price against house size for this location
            sizes = np.arange(500, 6001, 250)
            curve = pd.DataFrame([make_row(house_size_sqft=int(s)) for s in sizes])
            st.subheader('Price as the house gets bigger')
            st.line_chart(pd.Series(predict(curve, model_name) / 1e6, index=sizes, name='Rs million'), color=SAFFRON, height=240,
                          alt='Line chart of the estimated price in Rs million as house size grows from 500 to 6,000 sqft.')

        st.caption(f'Estimates are based on asking prices from ads, not final sale prices. Model: {model_label}.')

    # on a phone the card is far below the form, so the price stays pinned at the bottom
    st.markdown(f'<div class="price-bar"><span>Estimated asking price</span><b>{short(price)}</b></div>', unsafe_allow_html=True)

# =====================================================================
# tab 2: all models side by side
# =====================================================================
with tab_compare:
    st.subheader('The same house, four models')
    frame = pd.DataFrame([make_row()])
    estimates = {lab: float(predict(frame, MODELS[lab][0])[0]) for lab in MODELS}
    mid = float(np.median(list(estimates.values())))
    gap = (max(estimates.values()) - min(estimates.values())) / mid * 100
    st.markdown(f'<div class="verdict">For the house set on the first tab, the four models are within <b>{gap:.0f}%</b> '
                f'of each other. The model you picked there is the dark bar.</div>', unsafe_allow_html=True)
    st.write('')
    with st.container(key='card_models'):
        st.markdown('### Estimated asking price')
        model_bars(list(estimates.items()), model_label, short)

    st.subheader('How they scored on ads they never saw')
    results = pd.DataFrame(meta['results']).set_index('model')
    st.dataframe(results.sort_values('rmsle'), width='stretch', height=(len(results) + 1) * 35 + 3,
                 column_config={'rmsle': st.column_config.NumberColumn(format='%.3f'), 'r2 (log)': st.column_config.NumberColumn(format='%.3f'),
                                'mae (rs m)': st.column_config.NumberColumn(format='%.1f'), 'median abs % error': st.column_config.NumberColumn('median miss %', format='%.1f')})
    st.subheader('What each model is')
    rf_p = {k.split('__')[-1]: v for k, v in meta['best_params']['random forest'].items()}
    xgb_p = {k.split('__')[-1]: v for k, v in meta['best_params']['xgboost'].items()}
    INFO = {
        'Stacking': ('stacking (rf + xgb)',
                     'Takes the predictions of the random forest and xgboost, then a simple linear model learns how much to trust each.',
                     'Lowest error in the tests, and it is the default.',
                     'Only a little better than xgboost on its own, and it needs both models loaded.', []),
        'XGBoost': ('xgboost (tuned)',
                    'Builds trees one after another, each one fixing the mistakes of the ones before.',
                    'Best single model.',
                    'Many settings to tune and it needs enough ads per city.',
                    [('trees', xgb_p['n_estimators']), ('depth', xgb_p['max_depth']), ('learning rate', xgb_p['learning_rate'])]),
        'Random forest': ('random forest (tuned)',
                          'Many deep trees, each trained on a random sample of ads and features, then averaged.',
                          'Steady and hard to overfit. It has the smallest typical miss in percent before tuning.',
                          'Slightly worse than xgboost on the log price, and a big file.',
                          [('depth', rf_p['max_depth']), ('min leaf', rf_p['min_samples_leaf']), ('features per split', rf_p['max_features'])]),
        'Voting': ('voting (rf + xgb)',
                   'A plain average of the random forest and xgboost, no learning on top.',
                   'Simple, and it smooths out one model having a bad guess.',
                   'Treats both models as equally good.', []),
    }
    names = list(INFO)
    for row in range(0, len(names), 2):
        cols = st.columns(2, gap='large')
        for col, nm in zip(cols, names[row:row + 2]):
            key, how, good, watch, settings = INFO[nm]
            r = results.loc[key]
            stats = [('rmsle', f'{r["rmsle"]:.3f}'), ('typical miss', f'{r["median abs % error"]:.0f}%')] + [(k, v) for k, v in settings]
            with col:
                model_card(nm, nm == model_label, how, good, watch, stats)
        st.write('')

    st.write('')
    best = results['rmsle'].idxmin()
    st.markdown(f'<div class="verdict">Lowest error is <b>{best}</b> at {results["rmsle"].min():.3f} rmsle. '
                'Xgboost beats the random forest, and voting and stacking only add a little on top, '
                'so the choice of model matters less than the features.</div>', unsafe_allow_html=True)
    st.write('')
    st.caption('rmsle is the typical miss on the log price, lower is better. r2 is how much of the price spread the model explains. '
               'mae is the average miss in Rs million. All scored on the same 20% of ads kept aside.')

# =====================================================================
# tab 3: price map with timeline
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
        stats['elevation'] = np.minimum(stats[value_col] / np.percentile(stats[value_col], 95), 1.2) * 40000
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
        month_chart(pd.DataFrame(lines), 'Rs million',
                    'Line chart of the median asking price in Rs million for each month, for the whole island and the chosen districts.')
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
# tab 4: about
# =====================================================================
with tab_about:
    st.subheader('How the estimate is made')
    st.markdown(f"""
    The price comes from a pickled scikit-learn pipeline: preprocessing (imputing, scaling, one hot encoding of
    locations) followed by the model you pick on the first tab (stacking by default), trained on house ads from ikman.lk. The model predicts the log of the
    price, and the range shown is one test error either side (about 68% of ads fall inside it).

    - Reposted ads were removed before training so the test score is not flattering.
    - Prices are asking prices, not what the house finally sold for.
    - Ads cover Sep 2022 to Apr 2023 only, so the timeline is short and says nothing about longer trends.
    """)

    st.subheader('What drives the price')
    importance = pd.DataFrame(meta['importance']).sort_values('xgboost (tuned)', ascending=False)
    st.bar_chart(importance, horizontal=True, sort=False, stack=False, color=[GREEN, SAFFRON], height=330,
                 alt='Bar chart of how much each feature matters to the random forest and to xgboost.')
    st.caption('How much the error grows when a feature is shuffled, so longer means more important.')
