# %%
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import time
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# %% Load the data
print("Loading and Preprocessing Data")
data = pd.read_csv('./raw_data/usa-real-estate-dataset/realtor-data.zip.csv')
data.drop(['street', 'prev_sold_date', 'brokered_by'], axis=1, inplace=True)
data['zip_code'] = data['zip_code'].astype(str)
data.dropna(inplace=True)

# Initial data summary
print(f" Data Overview (n={len(data):,}):")
print(f" Price Range: ${data.price.min():,.0f} - ${data.price.max():,.0f}")
print(f"Features: {list(data.columns)}")

# %% train test split
X = data.drop(['price'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Log-transform numerical features and y_train
numerical_features = ['bed', 'bath', 'acre_lot', 'house_size']
print("Applying Log Transformations")
print(f"Transforming features: {numerical_features}")

X_train[numerical_features] = np.log1p(X_train[numerical_features])
X_test[numerical_features] = np.log1p(X_test[numerical_features])
y_train = np.log1p(y_train)

# %% Outlier Removal using IQR
def remove_outliers_iqr(df, column, iqr_factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR    
    return df[column].clip(lower_bound, upper_bound)

print("Outlier Removal Report")
for col in numerical_features:
    X_train[col] = remove_outliers_iqr(X_train, col)
    X_test[col] = remove_outliers_iqr(X_test, col)

# Visualize Outlier Impact
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for idx, col in enumerate(numerical_features):
    # Before Processing
    sns.boxplot(x=data[col], ax=axes[0, idx], color="salmon")
    axes[0, idx].set_title(f"Original {col}", fontsize=10)

    # After Processing
    sns.boxplot(x=X_train[col], ax=axes[1, idx], color="lightgreen")
    axes[1, idx].set_title(f"Processed {col}", fontsize=10)

# Transform target variables
y_train = remove_outliers_iqr(y_train.to_frame(), 'price').squeeze()
y_test = remove_outliers_iqr(y_test.to_frame(), 'price').squeeze()

# Visualize distributions
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
for idx, col in enumerate(numerical_features[:4]):
    sns.histplot(X_train[col], ax=axes[idx], kde=True)  # Access axes[idx] directly
    axes[idx].set_title(f'{col} Distribution')

plt.suptitle('Feature Distributions After Processing', y=1.02)
plt.tight_layout()
plt.show()
plt.close()

# %% Prepare data for modeling
high_cardinality_features = ['city', 'zip_code', 'status']
low_cardinality_features = ['state']

print("Feature Encoding Summary")
print(f"Target Encoded (high cardinality): {high_cardinality_features}")
print(f"One-Hot Encoded (low cardinality): {low_cardinality_features}")

encoder = TargetEncoder(target_type='continuous', smooth="auto")
X_train[high_cardinality_features] = encoder.fit_transform(X_train[high_cardinality_features], y_train)
X_test[high_cardinality_features] = encoder.transform(X_test[high_cardinality_features])

X_train = pd.get_dummies(X_train, columns=low_cardinality_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_cardinality_features, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# %% Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Model Comparison

fitted_models = {}

def evaluate_model(name, model):
    start_time = time.time()
    try:
        current_model = clone(model)
        current_model.fit(X_train_scaled, y_train)
        
        # Store fitted model
        fitted_models[name] = current_model
        
        y_pred = current_model.predict(X_test_scaled)
        y_pred_exp = np.expm1(y_pred).clip(1e4, 2e6)
        
        return {
            'Model': name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_exp)),
            'R2': r2_score(y_test, y_pred_exp),
            'Time (s)': time.time() - start_time
        }
    except Exception as e:
        print(f"⚠️ {name} failed: {str(e)}")
        return None

# Then run comparison
models = {
    'Lasso': Lasso(alpha=0.1, selection='random', max_iter=1000),
    'XGBoost': XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1,
                           tree_method='hist', n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5,
                                                 learning_rate=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10,
                                         min_samples_leaf=5, n_jobs=-1, random_state=42),
    'Ridge': Ridge(alpha=1.0, solver='sparse_cg'),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_split=20, random_state=42)
}

print("Training Models...")
with joblib.parallel_backend('threading', n_jobs=4):
    results = joblib.Parallel()(joblib.delayed(evaluate_model)(name, model) for name, model in models.items())

results_df = pd.DataFrame([r for r in results if r]).sort_values('RMSE')

# Display clean results table
print("\nModel Performance Comparison:")
results_df[['Model', 'RMSE', 'R2', 'Time (s)']].to_string(index=False, float_format="%.2f")
# Lasso              |   0.69
# XGBoost            |   0.61
# Ridge              |   0.61
# Gradient Boosting  |   0.61
# Random Forest      |   0.58
# Decision Tree      |   0.55

# %% residuals graph

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name].fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
y_pred_exp = np.expm1(y_pred).clip(1e4, 2e6)
residuals = y_test - y_pred_exp
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