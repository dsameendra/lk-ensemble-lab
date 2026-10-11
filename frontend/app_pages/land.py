import pickle
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from shared import (ROOT, TIMELINE, TIMELINE_LABELS, GREEN, SAFFRON, TOOLTIP_STYLE,
                    colour_scale, hero, inject_css, load_json, model_bars, model_card, month_chart, short)

inject_css()

TREES = ['rf', 'xgb', 'lgbm', 'cat']
MODELS = {
    'Stacking': ('stacking', 'Learns how much to trust each model, with ridge and a map based model added. Lowest error in the tests.'),
    'XGBoost': ('xgboost', 'Boosted trees that fix each other\'s mistakes. Best single model.'),
    'LightGBM': ('lightgbm', 'Fast boosted trees that read location as a category.'),
    'CatBoost': ('catboost', 'Boosted trees with built in handling of categories.'),
    'Random forest': ('random forest', 'Many deep trees averaged.'),
    'Voting': ('voting', 'Plain average of the four tree models.'),
}
RESULT_ROW = {'stacking': 'stacking (trees + ridge + knn)', 'xgboost': 'xgboost (tuned)', 'lightgbm': 'lightgbm (tuned)',
              'catboost': 'catboost (tuned)', 'random forest': 'random forest (tuned)', 'voting': 'voting (4 trees)'}
BASE_NAME = {'xgboost': 'xgb', 'lightgbm': 'lgbm', 'catboost': 'cat', 'random forest': 'rf'}
WORDS = {'main_road': 'Main road', 'highway': 'Near highway', 'clear_deed': 'Clear deed', 'near_town': 'Near town',
         'bare_land': 'Bare land', 'water': 'Water', 'electricity': 'Electricity', 'lake_or_beach': 'Lake or beach',
         'paddy': 'Paddy', 'investment': 'Investment'}
USES = ['Residential', 'Commercial', 'Agricultural']


# ---------- loading ----------
@st.cache_resource
def load_model():
    with open(ROOT / 'model' / 'land_stacking.pkl', 'rb') as f:
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
districts = sorted(meta['locations'])
coord_of = land.drop_duplicates('location').set_index('location')[['latitude', 'longitude']]
DEFAULTS = {'land_district': areas[0][0], 'land_location': areas[0][1], 'land_size': 10.0, 'land_use': ['Residential'], 'land_words': [],
            'land_membership': 'free', 'land_member': 'No', 'land_verified': 'No'}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_sample():
    f = sample['features']
    st.session_state.update({
        'land_district': f['district'], 'land_location': f['location'], 'land_size': float(f['land_size_perches']),
        'land_use': [u for u in USES if f[u.lower()]],
        'land_words': [WORDS[k] for k in WORDS if f[k]],
        'land_membership': f['membership_level'],
        'land_member': 'Yes' if f['is_member'] else 'No',
        'land_verified': 'Yes' if f['is_verified'] else 'No',
        'land_sample': True,
    })


def reset_flag():
    st.session_state['land_sample'] = False


def district_changed():
    # a new district starts on its busiest city
    d = st.session_state['land_district']
    st.session_state['land_location'] = max(meta['locations'][d], key=lambda l: ad_counts.get((d, l), 0))
    reset_flag()


def map_clicked():
    picked = st.session_state['land_map'].selection.objects.get('places', [])
    if picked:
        st.session_state['land_district'] = picked[0]['district']
        st.session_state['land_location'] = picked[0]['location']
        reset_flag()


# ---------- header ----------
hero('What is a plot of land worth, per perch?',
     'Pick an area, say how big the plot is and what the ad says about it. Perch gives a price per perch and a total, '
     'and shows how six models and four simple baselines compare on it.',
     [f'{len(land):,} land ads', '6 models', 'Sep 2022 to Apr 2023'], current='land')

tab_est, tab_compare, tab_map, tab_about = st.tabs(['Estimate a price', 'Compare models', 'Price map', 'About the model'])

# =====================================================================
# tab 1: estimate
# =====================================================================
with tab_est:
    left, right = st.columns([2, 3], gap='large')

    with left:
        with st.container(key='card_form'):
            st.markdown('### Describe the plot')
            c1, c2 = st.columns(2)
            district = c1.selectbox('District', districts, key='land_district', on_change=district_changed)
            location = c2.selectbox('City', sorted(meta['locations'][district]), key='land_location', on_change=reset_flag,
                                    help='Type to search. Clicking a dot on the map sets both.')
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
    typical_miss = next(r['median abs % error'] for r in meta['results'] if r['model'] == RESULT_ROW[model_name])
    spread = np.exp(meta['test_rmsle'][model_name])
    low, high = price / spread, price * spread

    quantiles = meta['price_quantiles'].get(district)
    pct = float(np.interp(price, quantiles, np.arange(101))) if quantiles else None
    pos = min(max(pct, 2), 98) if quantiles else 50
    last_pos = st.session_state.get('land_shown', pos)
    st.session_state['land_shown'] = pos

    with left:
        if quantiles:
            st.markdown(f"""
            <div class="card" style="margin-top: .2rem">
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

    with right:
        st.markdown(f"""
        <div class="card est" id="result" role="status">
          <div class="label">Estimated price per perch</div>
          <div class="big">{rs(price)}</div>
          <div class="sub">{size:g} perches comes to {short(price * size)} <span class="by">by {label}</span></div>
          <div class="range">Likely range: {rs(low)} to {rs(high)} per perch<br>
          <span class="note">About 2 in 3 ads like this one fall inside it.</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('land_sample'):
            actual = sample['actual_price_per_perch']
            err = abs(price - actual) / actual * 100
            st.markdown(f"""
            <div class="check"><b>A real listing the model never saw.</b>
            The real asking price was {rs(actual)} per perch, so the estimate is off by {err:.1f}%.
            One listing is not a score: over the whole test set the typical miss of {label} is {typical_miss:.0f}%.</div>
            """, unsafe_allow_html=True)

        st.write('')
        with st.container(key='card_map'):
            st.markdown('### Pick on the map')
            st.caption('Click a dot to set the district and city. Bigger dots have more ads, darker dots cost more.')
            places = area_stats(land)
            places = places[[(d, l) in known for d, l in zip(places['district'], places['location'])]].copy()
            places['radius'] = 900 + np.sqrt(places['ads']) * 120
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
                            on_select=map_clicked, selection_mode='single-object', key='land_map', height=560,
                            alt='Map of Sri Lanka with a dot for every area that has land ads. Bigger dots have more ads and darker dots have higher median prices per perch. The district and city lists make the same choice.')

    extras = st.container()
    st.markdown(f'<div class="price-bar"><span>Price per perch</span><b>{rs(price)}</b></div>', unsafe_allow_html=True)
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
# tab 2: all models side by side
# =====================================================================
with tab_compare:
    st.subheader('The same plot, six models')
    frame = pd.DataFrame([make_row()])
    estimates = {lab: float(predict(frame, MODELS[lab][0])[0]) for lab in MODELS}
    mid = float(np.median(list(estimates.values())))
    gap = (max(estimates.values()) - min(estimates.values())) / mid * 100
    st.markdown(f'<div class="verdict">For the plot set on the first tab, the six models are within <b>{gap:.0f}%</b> '
                f'of each other. The model you picked there is the dark bar.</div>', unsafe_allow_html=True)
    st.write('')
    with st.container(key='card_models'):
        st.markdown('### Estimated price per perch')
        model_bars(list(estimates.items()), label, rs)

    results = pd.DataFrame(meta['results']).set_index('model')
    kind = pd.Series('single tree model', index=results.index)
    kind[results.index.str.startswith(('district', 'location', 'ridge', 'knn'))] = 'simple baseline'
    kind[results.index.str.startswith(('voting', 'stacking'))] = 'ensemble'
    ranked = results[~results.index.str.contains('default|stacking \\(4')].sort_values('median abs % error', ascending=False)

    st.subheader('How they scored on ads they never saw')
    with st.container(key='card_scores'):
        st.markdown('### Typical miss on the test set')
        st.caption('Median absolute error as a share of the real price. Shorter is better.')
        model_bars([(m, v) for m, v in ranked['median abs % error'].items()], None, lambda v: f'{v:.1f}%')

    table = results.assign(type=kind).sort_values('rmsle')
    st.dataframe(table[['type', 'rmsle', 'r2 (log)', 'mae (rs k)', 'median abs % error']], width='stretch', height=(len(table) + 1) * 35 + 3,
                 column_config={'rmsle': st.column_config.NumberColumn(format='%.3f'), 'r2 (log)': st.column_config.NumberColumn(format='%.3f'),
                                'mae (rs k)': st.column_config.NumberColumn(format='%.0f'), 'median abs % error': st.column_config.NumberColumn('median miss %', format='%.1f')})

    st.subheader('What each model is')
    INFO = {
        'Stacking': ('stacking (trees + ridge + knn)',
                     'Takes the predictions of the four tree models, a ridge regression and a nearest neighbour model on the map, then a simple linear model learns how much to trust each.',
                     'Lowest error overall, and it keeps working if one model is off for a plot.',
                     'Only about 0.001 rmsle better than xgboost, which is inside the noise. It is also the biggest and slowest to load.'),
        'XGBoost': ('xgboost (tuned)',
                    'Builds trees one after another, each one fixing the mistakes of the ones before. Cities are one hot encoded and rare ones are grouped.',
                    'Best single model, and the one the shap importance and the leave-out check are run on.',
                    'Many settings to tune, and it can lean on the main cities when a city has few ads.'),
        'LightGBM': ('lightgbm (tuned)',
                     'Boosted trees grown leaf by leaf. It reads district, city and membership as categories directly instead of one hot columns.',
                     'Fast to train and as accurate as the forest.',
                     'A city with only a handful of ads can get a split to itself, so small cities are noisy.'),
        'CatBoost': ('catboost (tuned)',
                     'Boosted trees with its own handling of categories. It turns each category into a number from the prices seen before it in a shuffled order.',
                     'Steady with many cities and little tuning needed.',
                     'Slowest to train, and its error is the same as the others here.'),
        'Random forest': ('random forest (tuned)',
                          'Many deep trees, each trained on a random sample of ads and features, then averaged.',
                          'Hard to overfit, and it has the lowest typical miss in percent before ensembling.',
                          'Large file, and slightly worse than boosting on the log price.'),
        'Voting': ('voting (4 trees)',
                   'A plain average of the four tree models, no learning on top.',
                   'Simple, and smooths out the odd bad guess from one model.',
                   'Treats all four as equally good, which the stack weights show they are not.'),
    }
    weight_of = {'Stacking': None, 'XGBoost': 'xgb', 'LightGBM': 'lgbm', 'CatBoost': 'cat', 'Random forest': 'rf', 'Voting': None}
    names = list(INFO)
    for row in range(0, len(names), 2):
        cols = st.columns(2, gap='large')
        for col, nm in zip(cols, names[row:row + 2]):
            key, how, good, watch = INFO[nm]
            r = results.loc[key]
            stats = [('rmsle', f'{r["rmsle"]:.3f}'), ('typical miss', f'{r["median abs % error"]:.0f}%')]
            if weight_of[nm]:
                stats.append(('weight in stack', f'{meta["stack_weights"][weight_of[nm]]:.2f}'))
            with col:
                model_card(nm, nm == label, how, good, watch, stats)
        st.write('')

    base = results.loc[results.index.str.startswith(('district', 'location', 'ridge', 'knn')), 'rmsle'].min()
    trees = results.loc[~results.index.str.startswith(('district', 'location', 'ridge', 'knn', 'voting', 'stacking')), 'rmsle']
    st.write('')
    st.markdown(f'<div class="verdict">Every tree model beats the best simple baseline ({base:.3f} rmsle) by a clear margin, '
                f'the best at {trees.min():.3f}. The ensembles then add almost nothing: the stack is '
                f'{trees.min() - results.loc["stacking (trees + ridge + knn)", "rmsle"]:.3f} rmsle better than the best single model, '
                'which is inside the noise between runs. A typical miss stays around 32% because the ads say little about the plot itself.</div>',
                unsafe_allow_html=True)
    st.write('')
    st.caption('rmsle is the typical miss on the log price, lower is better. r2 is how much of the price spread the model explains. '
               'mae is in Rs thousand per perch. All scored on the same 20% of ads kept aside.')

# =====================================================================
# tab 3: price map with timeline
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
        stats['elevation'] = np.minimum(stats[value_col] / np.percentile(stats[value_col], 95), 1.2) * 40000
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
        month_chart(pd.DataFrame(lines), 'Rs thousand per perch',
                    'Line chart of the median asking price per perch in Rs thousand for each month, for the whole island and the chosen districts.')
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
# tab 4: about
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

    c1, c2 = st.columns(2, gap='large')
    with c1:
        st.subheader('What drives the price')
        importance = pd.Series(meta['importance']).sort_values(ascending=False)
        st.bar_chart(importance, horizontal=True, sort=False, color=GREEN, height=420,
                     alt='Bar chart of the average effect of each feature on the predicted log price per perch.')
        st.caption('Average shap value for xgboost, longer means the feature moves the price more.')
    with c2:
        st.subheader('Leaving features out')
        st.dataframe(pd.DataFrame(meta['ablation']).set_index('what is left out').round(3), width='stretch')
        st.caption('xgboost with one group of features removed. Location and coordinates carry the same information, so removing one costs little but removing both costs a lot.')
        st.subheader('Weights inside the stack')
        weights = pd.Series(meta['stack_weights'], name='weight').round(3)
        st.dataframe(weights.to_frame(), width='stretch')
        st.caption('rf, xgb, lgbm and cat are the four tree models. Ridge gets no weight, the map based knn a small one.')

