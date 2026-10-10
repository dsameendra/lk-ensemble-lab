# perch: frontend spec

streamlit app that loads the trained pipeline (`model/house_price_model.pkl`) and lets a user estimate the asking price of a house in sri lanka, and explore prices on a map.

## goals
- estimate a price from the details a user enters, with a range so it is not read as exact
- pick the location on a map instead of only a dropdown
- show how asking prices differ by area and over the months
- look like a small website, not a default streamlit page

## pages (tabs)
1. **estimate a price**
   - map of areas (dot size = number of ads, colour = median price), click to choose the location
   - form: district, location, bedrooms, bathrooms, house size, land size, seller details (membership, member, verified, month posted)
   - result: price in rupees and short form, likely range (estimate times and divided by exp(test rmsle)), position against district listings, price per sqft and per perch
   - extras from the model: same house in other busy areas, price as house size grows
   - button to load the held out listing and compare with its real price
2. **price map**
   - metric: median price, price per sqft, number of ads
   - style: 3d columns, heatmap, bubbles
   - filters: bedrooms range, timeline slider (oct 2022 to mar 2023) with a play button
   - kpis, median price by month for chosen districts, top and bottom areas
3. **about the model**
   - how the estimate is made, model comparison table, feature importance, caveats

## inputs (match the training columns)
bedrooms, bathrooms, house_size_sqft, land_size_perches, posted_month, is_verified, location, district, membership_level, is_member

## behaviour
- model and data loaded once and cached
- location options depend on the district, rare locations are not offered
- clicking a map dot sets district and location in the form
- months with too few ads are left out of the timeline

## look
- custom css, wide layout, default menu and footer hidden
- deep green header with a saffron accent, white cards, one font pair
- layout follows the design mock

## acceptance checks
- [x] app starts with `streamlit run frontend/app.py`
- [x] changing a field changes the prediction
- [x] held out listing gives the same prediction as the notebook
- [ ] checked on a phone width
