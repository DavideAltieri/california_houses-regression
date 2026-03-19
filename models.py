from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from xgboost import XGBRegressor

def build_linear_regression_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def build_random_forest(random_state: int = 42):
    return RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )

def build_xgboost(random_state: int = 42):
    return XGBRegressor(
        random_state=random_state,
        n_jobs=-1
    )
    
def get_models(random_state: int = 42):
    return {
        "Linear Regression": build_linear_regression_pipeline(),
        "Random Forest": build_random_forest(random_state=random_state),
        "XGBoost": build_xgboost(random_state=random_state),
    }

def get_cv(n_splits: int = 5, random_state: int = 42):
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def run_cross_validation(model, X, y, cv):
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    return cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

def tune_xgboost(X_train, y_train, random_state: int = 42):
    model = XGBRegressor(random_state=random_state, n_jobs=-1)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search

def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model