# perch: frontend spec

streamlit app with two pages, houses and land. it loads the trained pipelines in `model/` and lets a user estimate an asking price, and explore prices on a map.

## goals
- estimate a price from the details a user enters, with a range so it is not read as exact
- pick the area on a map instead of only a dropdown
- choose which model gives the estimate
- show how asking prices differ by area and over the months
- look like a small website, not a default streamlit page

## structure
- `app.py`: sets the page config and the navigation, nothing else
- `shared.py`: css, top bar with the page links, small helpers
- `app_pages/houses.py`: house prices (random forest, xgboost, stacking, voting)
- `app_pages/land.py`: land price per perch (stacking, voting, xgboost, lightgbm, catboost, random forest)

## each page has four tabs
1. **estimate a price**
   - form on the left with district and city lists, map on the right with plenty of height (dot size = number of ads, colour = median price), click a dot to choose the area
   - model chooser with its test error, result card with a likely range
     (estimate times and divided by exp(test rmsle), about 68% of ads), position against the district, small tiles
   - extras from the model: the same house or plot in other busy areas, price as size grows
   - button to load a real listing the model never saw and compare with its real price
   - why this estimate: the biggest pushes up and down for the house or plot as bars in percent, from the xgboost contributions (same chart on both pages, labelled as xgboost even when another model is chosen)
2. **compare models**
   - the same house or plot priced by every model, with how far apart they are
   - typical miss of every model and baseline, full score table
   - a card per model: how it works, good at, watch out, its scores
3. **price map**
   - metric, map style (3d columns, heatmap, bubbles), filters, a reset view button, all months combined
   - kpis, median price by month for chosen districts, top and bottom areas
4. **about the model**
   - how the estimate is made, feature importance, caveats
   - land page also shows the leave-features-out table and the stack weights

## inputs
- houses: bedrooms, bathrooms, house size, land size, area, membership, member, verified, month posted
- land: land size (perches), area, land use (residential, commercial, agricultural), words from the ad (main road, near highway, clear deed, near town, bare land, water, electricity, lake or beach, paddy, investment), membership, member, verified

## behaviour
- models and data are loaded once and cached
- the chosen model changes the estimate, the range, the area comparison and the size curve
- voting is the mean of the tree models on the log price, computed from the base models inside the saved stack
- area options only include places with enough ads for the model
- clicking a map dot sets the district and city in the form, changing the district resets the city
- months with too few ads are left out of the trend chart
- soft warnings for unusual input, the estimate still shows

## look
- custom css, wide layout, default menu and footer hidden
- deep green top bar and header in one band, saffron accents and active page link, white cards, one font pair
- layout follows the design mock

## acceptance checks
- [x] app starts with `streamlit run frontend/app.py`
- [x] changing a field changes the prediction
- [x] each model gives a different estimate
- [x] held out listings give the same prediction as the notebooks (houses and land)
- [x] checked on a phone width, no sideways scroll
- [x] months on the trend chart are in time order
- [ ] the xgboost base value plus the contributions equals its raw output for the same row
- [ ] why-this-estimate bars change when a field changes
