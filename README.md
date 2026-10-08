# lk-ensemble-lab

ensemble learning lab work on a sri lankan dataset. it compares random forest and xgboost for predicting house prices.

## problem

predict the asking price (rs) of a house for sale in sri lanka from its bedrooms, bathrooms, house size, land size, location and a few seller details. this is a regression problem.

## data

- house ads from ikman.lk, [sri lanka property ads dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset) on kaggle by ivantha. covers sep 2022 to apr 2023. the ad content belongs to the original posters, so the repo only keeps a cleaned numeric file

## setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
