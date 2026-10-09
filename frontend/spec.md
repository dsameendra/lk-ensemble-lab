# perch: frontend spec

streamlit app that loads the trained pipeline (`model/house_price_model.pkl`) and lets a user estimate the asking price of a house in sri lanka.

## goals
- estimate a price from the details a user enters
- show the price in rupees and in millions

## inputs (match the training columns)
bedrooms, bathrooms, house_size_sqft, land_size_perches, posted_month, is_verified, location, district, membership_level, is_member

## behaviour
- model loaded once and cached
- location options depend on the district

## acceptance checks
- [ ] app starts with `streamlit run frontend/app.py`
- [ ] changing a field changes the prediction
