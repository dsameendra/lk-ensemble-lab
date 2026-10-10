# frontend prompts

prompts i gave a genai tool to build the app. the data and model are mine, the tool only wrote the interface code around them.

## 1. first prompt

i trained a model that predicts the asking price of a house in sri lanka. it is a scikit-learn pipeline saved with pickle (`model/stacking.pkl`), so it takes raw values and handles the preprocessing itself.

build a small streamlit app that loads the pickle and shows text boxes and dropdowns for these fields, then shows the predicted price:

- bedrooms (1 to 10), bathrooms (1 to 10)
- house_size_sqft (200 to 20000), land_size_perches (1 to 500)
- district and location (location options depend on the district)
- membership_level (free, plus, premium), is_member, is_verified, posted_month

the model returns the price in rupees. show it with commas and also in millions.

## 2. second prompt

make it look like a proper website. custom css, a deep green header, saffron accent, white cards, nice fonts. add a likely price range using the test error of the model, and a button that loads a listing the model never saw and compares the prediction with the real price.

## 3. third prompt

add a map. users should be able to click an area to choose the location. add a second page with a price map across the island (3d columns, heatmap, bubbles), a timeline slider with a play button, and a chart of median price by month. add a page about the model with the comparison table and feature importance.

## 4. fourth prompt

i have three saved pipelines (random forest, xgboost, stacking) and voting can be worked out from the first two. let the user choose which model gives the estimate. show the test error of the chosen model and use it for the price range.
