import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# the saved land model uses the wrappers in estimators.py
sys.path.append(str(ROOT))

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# ads in the data run from sep 2022 to apr 2023, the first and last months are too thin to use
TIMELINE = [(2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2), (2023, 3)]
TIMELINE_LABELS = [f'{MONTHS[m - 1]} {y}' for y, m in TIMELINE]
GREEN, SAFFRON = '#0F5C50', '#F2B84B'

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700&family=Sora:wght@600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Manrope', sans-serif; }
span[translate="no"] { font-family: 'Material Symbols Rounded' !important; }

/* notes under the form */
[data-testid="stAlert"] { border-radius: 12px; border-left: 5px solid #F2B84B; }
[data-testid="stAlertContainer"] { background: #FFF6E0; color: #5C4410; padding: .7rem 1rem; }
[data-testid="stAlertContainer"] p { font-size: .9rem; line-height: 1.45; }
[data-testid="stAlertDynamicIcon"] { color: #B7791F; font-size: 1.25rem; }
.stApp { background: #F3F6F5; }
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 0 3rem 0; max-width: 100%; }

.hero h1 { font-family: 'Sora', sans-serif; font-size: 2.3rem; line-height: 1.15; margin: 1.6rem 0 .6rem 0; max-width: 40rem; color: #fff; text-wrap: balance; }
.hero p { color: #BFD6D0; font-size: 1.05rem; max-width: 38rem; line-height: 1.55; margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.3rem; }
.chips span { padding: .3rem .8rem; border-radius: 99px; background: rgba(255, 255, 255, .1); color: #E4F0ED; font-size: .82rem; font-weight: 700; }

/* the top bar and the intro are one dark band with no gaps */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-hero { background: #0F3D36; padding: 1rem 8vw 2.2rem 8vw; }
.st-key-hero [data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stTabs"] { padding: 0 8vw; margin-top: 1rem; }
[data-testid="stTabs"] [data-testid="stTabPanel"] { padding-top: 1rem; }
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
.by { font-size: .75rem; font-weight: 700; padding: .1rem .5rem; margin-left: .3rem; border-radius: 99px; background: #EAF3F0; color: #0F5C50; }
.note { font-size: .8rem; color: #4C625D; line-height: 1.5; }
.check { margin-top: .8rem; padding: .7rem .9rem; border-radius: 10px; background: #FFF6E0; font-size: .9rem; }

.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin-bottom: 1rem; }
.kpi { background: #fff; border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(16, 35, 31, .08); }
.kpi .v { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; }
.kpi .s { font-size: .8rem; color: #4C625D; }

button[data-baseweb="tab"] { font-weight: 700; font-size: 1rem; padding: .7rem 1.1rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0F5C50; }
[data-baseweb="tab-highlight"] { background: #F2B84B; height: 3px; }
div[data-testid="stButton"] button { border-radius: 12px; height: 2.8rem; font-weight: 700; border: 1px solid #C9D6D2; }
div[data-testid="stButton"] button:hover { border-color: #0F5C50; color: #0F5C50; }

/* numbers line up and the browser bits follow the palette */
.big, .sub, .tile .v, .kpi .v, .range, .bar-labels { font-variant-numeric: tabular-nums; }
::selection { background: #F2B84B; color: #10231F; }
:focus-visible { outline: 3px solid #F2B84B; outline-offset: 2px; }


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
    .st-key-hero { padding: 1rem 1.2rem 1.6rem 1.2rem; }
    .hero h1 { font-size: 1.6rem; }
    [data-testid="stTabs"] { padding: 0 1rem; }
    .big { font-size: 2rem; }
    .kpis { grid-template-columns: 1fr 1fr; }
    .block-container { padding-bottom: 5rem; }
    [class*="st-key-card_map"] [data-testid="stDeckGlJsonChart"] { height: 320px !important; }
    .price-bar { display: flex; justify-content: space-between; align-items: center; position: fixed; left: 0; right: 0; bottom: 0;
                 z-index: 1000; padding: .8rem 1.2rem; background: #0F3D36; color: #BFD6D0; font-size: .85rem;
                 box-shadow: 0 -6px 20px rgba(16, 35, 31, .25); font-variant-numeric: tabular-nums; }
    .price-bar b { font-family: 'Sora', sans-serif; font-size: 1.3rem; color: #F2B84B; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* top bar with the brand and the page links */
.st-key-hero .brand { display: flex; align-items: center; gap: .7rem; }
.st-key-hero .logo { width: 38px; height: 38px; border-radius: 10px; background: #F2B84B; color: #0F3D36;
                     display: grid; place-items: center; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.3rem; }
.st-key-hero .name { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; letter-spacing: .5px; color: #fff; }
.st-key-hero [data-testid="stPageLink-NavLink"] { color: #E4F0ED; border-radius: 10px; justify-content: center; background: transparent; }
.st-key-hero [data-testid="stPageLink-NavLink"] * { color: inherit; font-weight: 700; }
.st-key-hero [data-testid="stPageLink-NavLink"]:hover { background: rgba(255, 255, 255, .1); }
.st-key-hero [data-testid="stPageLink-NavLink"][disabled] { background: #F2B84B; color: #0F3D36; opacity: 1; }

/* model comparison */
.mcard { background: #fff; border-radius: 14px; padding: 1.1rem 1.2rem; box-shadow: 0 4px 16px rgba(16, 35, 31, .08); border-top: 4px solid #CFE5DF; height: 100%; }
.mcard.on { border-top-color: #F2B84B; }
.mcard h4 { font-family: 'Sora', sans-serif; font-size: 1.05rem; margin: 0 0 .5rem 0; display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.mcard .pick { font-family: 'Manrope', sans-serif; font-size: .7rem; padding: .1rem .5rem; border-radius: 99px; background: #FFF6E0; color: #7A5B12; }
.mcard .stats { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }
.mcard .stats span { font-size: .78rem; padding: .15rem .55rem; border-radius: 8px; background: #EAF3F0; color: #4C625D; }
.mcard .stats b { color: #10231F; }
.mcard p { font-size: .88rem; line-height: 1.5; margin: .35rem 0; color: #2A3F3A; }
.verdict { padding: .9rem 1.1rem; border-radius: 12px; background: #EAF3F0; border-left: 5px solid #0F5C50; line-height: 1.55; }
.model-row { display: grid; grid-template-columns: 13rem 1fr 7rem; align-items: center; gap: .8rem; padding: .35rem 0; }
.model-row .nm { font-weight: 700; font-size: .92rem; }
.model-row .track { height: 14px; background: #E4EDEA; border-radius: 7px; position: relative; }
.model-row .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; background: #9CC5BB; }
.model-row.picked .fill { background: #0F5C50; }
.model-row.picked .nm { color: #0F5C50; }
.model-row .val { text-align: right; font-variant-numeric: tabular-nums; font-size: .9rem; }
@media (max-width: 800px) { .model-row { grid-template-columns: 6.5rem 1fr 5.5rem; } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, text, chips, current):
    with st.container(key='hero'):
        brand, links = st.columns(2, vertical_alignment='center')
        brand.markdown('<div class="brand"><div class="logo">P</div><div class="name">Perch</div></div>', unsafe_allow_html=True)
        with links.container(horizontal=True, horizontal_alignment='right'):
            # the page you are on is the disabled link, it is styled as the active one
            st.page_link('app_pages/houses.py', label='Houses', icon=':material/home:', disabled=current == 'houses')
            st.page_link('app_pages/land.py', label='Land', icon=':material/landscape:', disabled=current == 'land')
        spans = ''.join(f'<span>{c}</span>' for c in chips)
        st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p><div class="chips">{spans}</div></div>', unsafe_allow_html=True)


def month_chart(frame, unit, description):
    # keeps the months in time order, the built in line chart sorts them by name
    long = frame.rename_axis('month').reset_index().melt('month', var_name='area', value_name=unit)
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X('month:N', sort=TIMELINE_LABELS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f'{unit}:Q', title=unit), color=alt.Color('area:N', title=None, legend=alt.Legend(orient='bottom')),
    ).properties(height=300)
    st.altair_chart(chart, width='stretch', alt=description)


def ebm_effects(pipe, row):
    # contribution of every term of the explainable boosting model to the log price, exact and additive
    ebm = pipe.named_steps['model'].regressor_
    return pd.Series(ebm.eval_terms(row)[0], index=ebm.term_names_)


def why_chart(effects, labels, description):
    # biggest pushes first, as a percent of the price: green raises it, saffron lowers it
    top = effects.reindex(effects.abs().sort_values(ascending=False).index)[:8]
    data = pd.DataFrame({'feature': [' x '.join(labels.get(part, part) for part in k.split(' & ')) for k in top.index], 'effect': (np.exp(top.values) - 1) * 100})
    data['direction'] = np.where(data['effect'] >= 0, 'raises the price', 'lowers the price')
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y('feature:N', sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=260)),
        x=alt.X('effect:Q', title='effect on the price (%)'),
        color=alt.Color('direction:N', title=None, scale=alt.Scale(domain=['raises the price', 'lowers the price'], range=[GREEN, SAFFRON]),
                        legend=alt.Legend(orient='bottom')),
        tooltip=['feature', alt.Tooltip('effect:Q', format='+.1f')],
    ).properties(height=34 * len(data))
    st.altair_chart(chart, width='stretch', alt=description)


def model_bars(rows, picked, fmt):
    # one bar per model, the one chosen on the estimate tab is dark
    top = max(v for _, v in rows)
    html = ''
    for name, value in rows:
        cls = ' picked' if name == picked else ''
        html += (f'<div class="model-row{cls}"><div class="nm">{name}</div>'
                 f'<div class="track"><div class="fill" style="width: {value / top * 100:.1f}%"></div></div>'
                 f'<div class="val">{fmt(value)}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def model_card(name, picked, how, good, watch, stats):
    chips = ''.join(f'<span><b>{v}</b> {k}</span>' for k, v in stats)
    mark = '<span class="pick">chosen on the first tab</span>' if picked else ''
    st.markdown(f'<div class="mcard{" on" if picked else ""}"><h4>{name}{mark}</h4><div class="stats">{chips}</div>'
                f'<p><b>How it works.</b> {how}</p><p><b>Good at.</b> {good}</p><p><b>Watch out.</b> {watch}</p></div>',
                unsafe_allow_html=True)


@st.cache_data
def load_json(name):
    with open(ROOT / 'model' / name) as f:
        return json.load(f)


def short(price):
    return f'Rs {price / 1e6:,.1f} M'


def colour_scale(values):
    # light green to deep green, scaled between the 5th and 95th percentile
    lo, hi = np.percentile(values, [5, 95])
    t = np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)
    light, dark = np.array([207, 229, 223]), np.array([15, 92, 80])
    return [[int(c) for c in light + (dark - light) * x] + [210] for x in t]


TOOLTIP_STYLE = {'backgroundColor': '#0F3D36', 'color': 'white', 'fontFamily': 'Manrope', 'fontSize': '13px'}
