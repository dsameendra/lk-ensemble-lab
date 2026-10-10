from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# small wrappers so lightgbm and catboost know which columns are categories without extra fit arguments.
# the saved model files use these classes, so the app imports this file too.


class LGBMWithCategories(LGBMRegressor):
    # the first three columns are the integer codes of location, district and membership
    def fit(self, X, y, **kwargs):
        return super().fit(X, y, categorical_feature=[0, 1, 2], **kwargs)


class CatBoostWithCategories(CatBoostRegressor):
    # the prepared frame still has these columns as text
    def fit(self, X, y, **kwargs):
        return super().fit(X, y, cat_features=['location', 'district', 'membership_level'], **kwargs)
