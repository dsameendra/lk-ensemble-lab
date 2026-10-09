# frontend prompts

prompts i gave a genai tool to build the app. the data and model are mine, the tool only wrote the interface code around them.

## 1. first prompt

i trained a model that predicts the asking price of a house in sri lanka. it is a scikit-learn pipeline saved with pickle (`model/house_price_model.pkl`), so it takes raw values and handles the preprocessing itself.

build a small streamlit app that loads the pickle and shows text boxes and dropdowns for these fields, then shows the predicted price:

- bedrooms (1 to 10), bathrooms (1 to 10)
- house_size_sqft (200 to 20000), land_size_perches (1 to 500)
- district and location (location options depend on the district)
- membership_level (free, plus, premium), is_member, is_verified, posted_month

the model returns the price in rupees. show it with commas and also in millions.
