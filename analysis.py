import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


def evaluate_usa_housing_dataset_with_states():
    """Let's test the model and see how it does. The results are as follows:

    Linear Regression Results:
    * MSE: 75679816566.33786
    * R^2: -0.039849538950929464

    Random Forest Results:
    * MSE: 73394453236.17204
    * R^2: -0.008448379262266714

    XGBoost Results:
    * MSE: 72991107572.15358
    * R^2: -0.0029063626216765392

    Pretty bad results. It was improved compared to the following previous results, but still did not significantly improve.

    Linear Regression Results:
    * MSE: 70597692136.68445
    * R^2: -0.033919157433573544

    Random Forest Results:
    * MSE: 71108152392.66861
    * R^2: -0.04139496325380687

    XGBoost
    * MSE: 80887792335.49518
    * R^2: -0.18462000055551275

    """

    # Load the data
    training_data = pd.read_csv('./model_data/training_usa_housing_dataset_with_states.csv')
    testing_data = pd.read_csv('./model_data/testing_usa_housing_dataset_with_states.csv')

    X_train_scaled = training_data.drop(['Price'], axis=1)
    y_train = training_data['Price']
    X_test_scaled = testing_data.drop(['Price'], axis=1)
    y_test = testing_data['Price']

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred_result = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, pred_result)
        r2 = r2_score(y_test, pred_result)
        print(f"{name} Results:")
        print(f"  MSE: {mse}")
        print(f"  R^2: {r2}")

evaluate_usa_housing_dataset_with_states()