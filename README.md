# lk-ensemble-lab

ensemble learning lab work on a sri lankan dataset. it compares random forest and xgboost for predicting house prices.

## problem

predict the asking price (rs) of a house for sale in sri lanka from its bedrooms, bathrooms, house size, land size, location and a few seller details. this is a regression problem.

## data

- house ads from ikman.lk, [sri lanka property ads dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset) on kaggle by ivantha. covers sep 2022 to apr 2023. the ad content belongs to the original posters, so the repo only keeps a cleaned numeric file

## results

scored on a 20% test set (rmsle on log price, lower is better):

| model | rmsle | r2 (log) | median error |
|---|---|---|---|
| random forest (default) | 0.392 | 0.780 | 15.4% |
| xgboost (default) | 0.389 | 0.783 | 18.8% |
| random forest (tuned) | 0.390 | 0.782 | 17.1% |
| xgboost (tuned) | 0.376 | 0.798 | 17.3% |
| voting (rf + xgb) | 0.376 | 0.798 | 17.0% |
| stacking (rf + xgb) | 0.374 | 0.800 | 17.0% |

xgboost beats the random forest, and voting and stacking add very little on top.

## setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

on a mac xgboost also needs openmp: `brew install libomp`.
