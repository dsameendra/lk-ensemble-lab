# lk-ensemble-lab

ensemble learning lab work on a sri lankan dataset. it compares random forest and xgboost for predicting house prices, adds a voting and a stacking model, runs inference on one held out listing and serves the model through a small web app called perch.

## problem

predict the asking price (rs) of a house for sale in sri lanka from its bedrooms, bathrooms, house size, land size, location and a few seller details. this is a regression problem.

## data

- house ads from ikman.lk, [sri lanka property ads dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset) on kaggle by ivantha. covers sep 2022 to apr 2023. the ad content belongs to the original posters, so the repo only keeps a cleaned numeric file
- location coordinates for the map are looked up with openstreetmap nominatim

## results

scored on a 20% test set (rmsle on log price, lower is better):

| model | rmsle | r2 (log) | median error |
|---|---|---|---|
| random forest (default) | 0.392 | 0.780 | 15.4% |
| xgboost (default) | 0.389 | 0.783 | 18.8% |
| random forest (tuned) | 0.390 | 0.782 | 17.1% |
| xgboost (tuned) | 0.377 | 0.797 | 17.7% |
| voting (rf + xgb) | 0.378 | 0.796 | 17.3% |
| stacking (rf + xgb) | 0.376 | 0.798 | 17.5% |

xgboost beats the random forest, and voting and stacking add very little on top. the app lets you choose between stacking (default), xgboost, random forest and voting. details, strengths, limitations and interpretation are in `notebooks/modelling.ipynb`.

## structure

```
data/raw         kagglehub downloads (not tracked)
data/processed   cleaned houses and location coordinates
notebooks/       eda, modelling, location coordinates
model/           pickled pipeline, held out sample, app metadata
frontend/        streamlit app (perch), spec and prompts
```

notebooks run in order: `eda`, `modelling`, `locations`.

## setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

on a mac xgboost also needs openmp: `brew install libomp`.

## run the app

```
streamlit run frontend/app.py
```

## limitations

- asking prices, not sale prices
- 7 months of ads, so the timeline says nothing about long term trends
- colombo and gampaha make up most of the data, small districts have large errors
