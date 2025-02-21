# %% 1. Initial Setup
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import CatBoostEncoder
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import joblib
import warnings
import shap

# Configuration
warnings.filterwarnings('ignore', category=UserWarning)

# %% 2. Data Loading & EDA
print("Loading and Exploring Data")
data = pd.read_csv('./raw_data/usa-real-estate-dataset/realtor-data.zip.csv')

# Initial Numerical Features Analysis
initial_features = ["bed", "bath", "acre_lot", "house_size", "price"]

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data[initial_features].corr(), annot=True, cmap="grey", fmt=".2f")
plt.title("Initial Feature Correlations", pad=20, fontsize=16)
plt.tight_layout()
plt.show()

# Feature Distributions
plt.figure(figsize=(15, 10))
for idx, feature in enumerate(initial_features, 1):
    plt.subplot(2, 3, idx)
    sns.histplot(data[feature], bins=50, kde=True)
    plt.title(f"{feature.title()} Distribution", fontsize=12)
    plt.xlabel("")
plt.suptitle("Initial Feature Distributions", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

data.head()
# %% 3. Data Cleaning
print("Cleaning Data")
data = data.drop(["street", "prev_sold_date", "brokered_by"], axis=1)
data["zip_code"] = data["zip_code"].astype(str)
data["price"] = data["price"].clip(lower=20_000, upper=375_000_000)
data = data.dropna()


print(f"Clean Data Overview (n={len(data):,}):")
print(f"Price Range: ${data.price.min():,.0f} - ${data.price.max():,.0f}")
print(f"Features: {list(data.columns)}")

data.info()
data.columns

# %% 4. Data Preprocessing

# Train-Test Split
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log Transformations
numerical_features = ["bed", "bath", "acre_lot", "house_size"]
print("Applying Log Transformations to:", numerical_features)

X_train[numerical_features] = np.log1p(X_train[numerical_features])
X_test[numerical_features] = np.log1p(X_test[numerical_features])
y_train = np.log1p(y_train)

# Outlier Handling
def remove_outliers_iqr(df, column, iqr_factor=1):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    bounds = (Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR)
    return df[column].clip(*bounds)

for col in numerical_features:
    X_train[col] = remove_outliers_iqr(X_train, col, iqr_factor=0.8)  
    X_test[col] = remove_outliers_iqr(X_test, col, iqr_factor=0.8)
    
# Visualize distributions
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  
for idx, col in enumerate(numerical_features[:4]):
    sns.histplot(X_train[col], ax=axes[idx], kde=True)  
    axes[idx].set_title(f'{col} Distribution')
plt.suptitle('Feature Distributions After Processing', y=1.02)
plt.tight_layout()
plt.show()
plt.close()


# Apply Outlier Removal
for col in numerical_features:
    X_train[col] = remove_outliers_iqr(X_train, col)
    X_test[col] = remove_outliers_iqr(X_test, col)

# Target clipping
y_train = remove_outliers_iqr(y_train.to_frame(), "price", iqr_factor=0.2).squeeze()  # From 0.2
y_test = remove_outliers_iqr(y_test.to_frame(), "price", iqr_factor=0.2).squeeze()

# Feature Engineering
# Target Encoding
high_cardinality = ['city', 'zip_code', 'status']
low_cardinality = ['state']

# Initialize encoder
te_encoder = TargetEncoder(target_type='continuous', smooth='auto')

# Apply encoding only to specified columns
X_train[high_cardinality] = te_encoder.fit_transform(X_train[high_cardinality], y_train)
X_test[high_cardinality] = te_encoder.transform(X_test[high_cardinality])

# Add Interaction Features
X_train['bed_bath_ratio'] = X_train['bed']**2 / (X_train['bath'] + 1e-6)
X_test['bed_bath_ratio'] = X_test['bed']**2 / (X_test['bath'] + 1e-6)

#Polynomial Features
X_train['house_size_squared'] = X_train['house_size']**2
X_test['house_size_squared'] = X_test['house_size']**2

X_train['house_size_x_zip'] = X_train['house_size'] * X_train['zip_code']
X_test['house_size_x_zip'] = X_test['house_size'] * X_test['zip_code']

# One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=low_cardinality, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_cardinality, drop_first=True)

# Column alignment
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# %% 5. Model Development 

selector = SelectFromModel(
    estimator=XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    ),
    threshold='1.25*median'
)

X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
X_test_reduced = selector.transform(X_test_scaled)

model_results = []
fitted_models = {}

def evaluate_model(name, model, param_grid=None):
    start_time = time.time()
    try:
        if param_grid:
            # Use RandomizedSearchCV instead of GridSearchCV
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10,  
                scoring='r2',
                cv=3,
                verbose=1,
                n_jobs=-1 
            )
            search.fit(X_train_scaled, y_train)
            current_model = search.best_estimator_
            print(f"Best Params for {name}: {search.best_params_}")
        else:
            current_model = clone(model)
            current_model.fit(X_train_scaled, y_train)

        # Store fitted model
        fitted_models[name] = current_model

        # Predictions and evaluation
        y_pred = current_model.predict(X_test_scaled)
        y_pred_exp = np.expm1(y_pred).clip(1e4, 2e6)  

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_exp))
        r2 = r2_score(y_test, y_pred_exp)

        print(f"{name}: RMSE={rmse:.2f}, R²={r2:.4f}, Time={time.time() - start_time:.2f}s")

        return {
            'Model': name,
            'RMSE': rmse,
            'R2': r2,
            'Time (s)': time.time() - start_time,
            'ModelInstance': current_model
        }
    except Exception as e:
        print(f"⚠️ {name} failed: {str(e)}")
        return None

# Optimized parameter grids
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [4, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'Gradient Boosting': {
        'max_iter': [150],
        'max_depth': [4, 6],
        'learning_rate': [0.08, 0.15],
        'min_samples_leaf': [10, 20]
    },
}

# Focus on top 3 performers only
models = {
    'Lasso': Lasso(alpha=0.05, max_iter=2000),
    'XGBoost': XGBRegressor(
        reg_alpha=0.5, reg_lambda=0.8, subsample=0.8, colsample_bytree=0.7),
    'Gradient Boosting': HistGradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12,
                                         min_samples_leaf=3, n_jobs=-1, random_state=42),
    'Ridge': Ridge(alpha=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
}

print("Training Optimized Models...")
with joblib.parallel_backend('loky', n_jobs=-1):
    results = joblib.Parallel()(
        joblib.delayed(evaluate_model)(
            name,
            model,
            param_grids.get(name)
        ) for name, model in models.items()
    )

# Populate fitted_models from results
for result in results:
    if result:
        fitted_models[result['Model']] = result['ModelInstance']



# %% 6. Stacking models

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# SHAP analysis for XGBoost
explainer = shap.TreeExplainer(fitted_models['XGBoost'])
shap_values = explainer.shap_values(X_test_scaled_df)
shap.summary_plot(shap_values, X_test_scaled_df, show=False)
plt.title("XGBoost Feature Impact (SHAP Values)", y=1.1)
plt.tight_layout()
plt.show()

stack = StackingRegressor(
    estimators=[
        ('xgb', fitted_models['XGBoost']),
        ('hgb', fitted_models['Gradient Boosting'])
    ],
    final_estimator=Ridge(),
    passthrough=True,
    n_jobs=-1
)

# Evaluate and add to results
stack_results = evaluate_model('Stacked', stack)
if stack_results:
    results.append(stack_results)
# Final results compilation

results_df = pd.DataFrame([r for r in results if r]).sort_values('RMSE')
results_df
# %% 7. Results Visualization

# Horizontal Bar Chart for Model Performance
plt.figure(figsize=(12, 8))
sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE"),
    hue="Model",
    palette="viridis",
)
plt.title("Model Performance: RMSE by Model", fontsize=16, pad=15)
plt.xlabel("Root Mean Squared Error (USD)", fontsize=14)
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Residual Analysis for Best Model
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name].fit(X_train_scaled, y_train)

# Predictions and residuals
y_pred = best_model.predict(X_test_scaled)
y_pred_exp = np.expm1(y_pred).clip(1e4, 2e6)
residuals = y_test - y_pred_exp

# Handle potential NaNs or infinities in residuals
residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

# Improved Residual Histogram
plt.figure(figsize=(14, 8))
sns.histplot(
    x=residuals,
    bins=80,
    kde=True,
    color="darkblue",
    line_kws={"linewidth": 2},
)
plt.title(f"Residual Distribution of Best Model ({best_model_name})", fontsize=16, pad=15)
plt.xlabel("Prediction Error (USD)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")
plt.text(
    x=residuals.mean(),
    y=plt.ylim()[1] * 0.9,
    s=f"Mean Error: ${residuals.mean():,.0f}",
    color="black",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
)
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
plt.scatter(y_test/1e6, residuals/1e6, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Price Prediction Residuals (Millions USD)", fontsize=16)
plt.xlabel("Actual Prices ($M)")
plt.ylabel("Prediction Errors ($M)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
