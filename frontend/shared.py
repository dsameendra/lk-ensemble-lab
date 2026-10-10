import json
import sys
from pathlib import Path

import numpy as np
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
span[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #F3F6F5; }
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 0 3rem 0; max-width: 100%; }

.hero { background: #0F3D36; padding: .8rem 8vw 5.5rem 8vw; color: #fff; }
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

/* top bar with the brand and the page links */
.st-key-topbar { background: #0F3D36; padding: 1rem 8vw 0 8vw; }
.st-key-topbar .brand { display: flex; align-items: center; gap: .7rem; }
.st-key-topbar .logo { width: 38px; height: 38px; border-radius: 10px; background: #F2B84B; color: #0F3D36;
                       display: grid; place-items: center; font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.3rem; }
.st-key-topbar .name { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; letter-spacing: .5px; color: #fff; }
.st-key-topbar [data-testid="stPageLink-NavLink"] { color: #BFD6D0; border-radius: 10px; font-weight: 700; justify-content: center; }
.st-key-topbar [data-testid="stPageLink-NavLink"]:hover { background: rgba(255, 255, 255, .08); }
.st-key-topbar [data-testid="stPageLink-NavLink"][aria-current="page"] { background: rgba(242, 184, 75, .18); color: #F2B84B; }
.st-key-topbar [data-testid="stPageLink-NavLink"] p { color: inherit; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def topbar():
    with st.container(key='topbar'):
        brand, houses, land = st.columns([6, 1.2, 1.2], vertical_alignment='center')
        brand.markdown('<div class="brand"><div class="logo">P</div><div class="name">Perch</div></div>', unsafe_allow_html=True)
        houses.page_link('app_pages/houses.py', label='Houses', icon=':material/home:')
        land.page_link('app_pages/land.py', label='Land', icon=':material/landscape:')


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
