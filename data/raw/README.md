# raw data

the raw files are not in the repo because they are too large for git (properties.csv alone is about 229 MB).

run `notebooks/houses_eda.ipynb` to download them with kagglehub. `notebooks/land_eda.ipynb` reuses the same download.

## source and data use

the files are the [sri lanka property ads dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset) by ivantha on kaggle, ads scraped from ikman.lk and shared under the odc attribution licence (odc-by). the ad content belongs to the original posters.

the cleaned files in `data/processed` keep only numbers and place names (price, sizes, rooms, location, district, seller flags). no names, phone numbers, ad titles, descriptions or seller ids are kept. the land file keeps the length of the description and ten yes or no flags for words in it, not the text.

