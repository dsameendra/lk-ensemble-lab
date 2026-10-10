# lk-ensemble-lab

ensemble learning lab work on a sri lankan dataset. part 1 compares random forest and xgboost for predicting house prices, adds a voting and a stacking model and runs inference on one held out listing. part 2 takes a second case from the same ads, land price per perch, and adds lightgbm, catboost, a bigger stack and a comparison with simple baselines. both are served through a small web app called perch.

## problem

part 1: predict the asking price (rs) of a house for sale in sri lanka from its bedrooms, bathrooms, house size, land size, location and a few seller details.

part 2: predict the asking price per perch (rs) of a plot of land from its size, land use, location, seller details and words used in the ad. both are regression problems.

## data

- house and land ads from ikman.lk, [sri lanka property ads dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset) on kaggle by ivantha. covers sep 2022 to apr 2023. the ad content belongs to the original posters, so the repo only keeps cleaned numeric files
- about 70% of land ads are reposts (same price, size and location), so they are removed before splitting to avoid leakage
- land prices come in different units, so everything is converted to rs per perch (per acre is divided by 160)
- location coordinates for the map are looked up with openstreetmap nominatim

## results: houses

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

## results: land

scored on a 20% test set (rmsle on log price per perch, lower is better):

| model | rmsle | r2 (log) | median error |
|---|---|---|---|
| district median | 1.083 | 0.308 | 58.7% |
| location median | 0.920 | 0.502 | 35.2% |
| ridge | 0.847 | 0.577 | 39.1% |
| knn (location + size) | 0.883 | 0.541 | 36.7% |
| random forest (tuned) | 0.778 | 0.644 | 31.5% |
| xgboost (tuned) | 0.769 | 0.651 | 32.3% |
| lightgbm (tuned) | 0.777 | 0.644 | 32.8% |
| catboost (tuned) | 0.778 | 0.644 | 33.1% |
| voting (4 trees) | 0.770 | 0.651 | 32.1% |
| stacking (trees + ridge + knn) | 0.769 | 0.652 | 31.8% |

all the tree models beat the baselines clearly, but they are very close to each other and the stack is only about 0.001 better than xgboost, which is inside the noise. a typical miss is still around 32%, because the ads say little about the plot itself. the app lets you choose between six land models. shap, a leave-features-out check and the stack weights are in `notebooks/land_models.ipynb`.

## structure

```
data/raw         kagglehub downloads (not tracked)
data/processed   cleaned houses, cleaned land and location coordinates
notebooks/       house eda and modelling, land eda and models, location coordinates
model/           pickled pipelines, held out samples, app metadata
estimators.py    lightgbm and catboost wrappers needed to unpickle the land model
frontend/        streamlit app (perch) with a houses and a land page, and its spec
```

notebooks run in order: `eda`, `modelling`, `locations`, `land_eda`, `land_models`.

## setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

on a mac xgboost and lightgbm also need openmp: `brew install libomp`.

## run the app

```
streamlit run frontend/app.py
```

## limitations

- asking prices, not sale prices
- 7 months of ads, so the timeline says nothing about long term trends
- colombo and gampaha make up most of the data, small districts have large errors
- the land model is about a third off on a typical plot, so treat it as a rough guide
- `model/land_model.pkl` is about 52 MB
