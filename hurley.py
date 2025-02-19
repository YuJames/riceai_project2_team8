# %% 1. Initial Setup
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import joblib
import warnings

# Configuration
warnings.filterwarnings('ignore', category=UserWarning)

# %% 2. Data Loading & EDA
print("Loading and Exploring Data")
data = pd.read_csv('./raw_data/usa-real-estate-dataset/realtor-data.zip.csv')

# Initial Numerical Features Analysis
initial_features = ['bed', 'bath', 'acre_lot', 'house_size', 'price']

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data[initial_features].corr(), annot=True, cmap='grey', fmt=".2f")
plt.title('Initial Feature Correlations', pad=20, fontsize=16)
plt.tight_layout()
plt.show()

# Feature Distributions
plt.figure(figsize=(15, 10))
for idx, feature in enumerate(initial_features, 1):
    plt.subplot(2, 3, idx)
    sns.histplot(data[feature], bins=50, kde=True)
    plt.title(f'{feature.title()} Distribution', fontsize=12)
    plt.xlabel('')
plt.suptitle('Initial Feature Distributions', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# %% 3. Data Cleaning
print("Cleaning Data")
data = data.drop(['street', 'prev_sold_date', 'brokered_by'], axis=1)
data['zip_code'] = data['zip_code'].astype(str)
data = data.dropna()

print(f"Clean Data Overview (n={len(data):,}):")
print(f"Price Range: ${data.price.min():,.0f} - ${data.price.max():,.0f}")
print(f"Features: {list(data.columns)}")

# %% 4. Data Preprocessing
# Train-Test Split
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log Transformations
numerical_features = ['bed', 'bath', 'acre_lot', 'house_size']
print("Applying Log Transformations to:", numerical_features)

X_train[numerical_features] = np.log1p(X_train[numerical_features])
X_test[numerical_features] = np.log1p(X_test[numerical_features])
y_train = np.log1p(y_train)

# Outlier Handling
def remove_outliers_iqr(df, column, iqr_factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    bounds = (Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR)
    return df[column].clip(*bounds)

# Visualize Outlier Impact
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for idx, col in enumerate(numerical_features):
    # Before Processing
    sns.boxplot(x=data[col], ax=axes[0, idx], color='salmon')
    axes[0, idx].set_title(f'Original {col}', fontsize=10)
    
    # After Processing
    sns.boxplot(x=X_train[col], ax=axes[1, idx], color='lightgreen')
    axes[1, idx].set_title(f'Processed {col}', fontsize=10)

plt.suptitle('Outlier Treatment Comparison', y=1.05, fontsize=14)
plt.tight_layout()
plt.show()

# Apply Outlier Removal
for col in numerical_features:
    X_train[col] = remove_outliers_iqr(X_train, col)
    X_test[col] = remove_outliers_iqr(X_test, col)

y_train = remove_outliers_iqr(y_train.to_frame(), 'price').squeeze()
y_test = remove_outliers_iqr(y_test.to_frame(), 'price').squeeze()

# %% 5. Feature Engineering
# Target Encoding
high_cardinality = ['city', 'zip_code', 'status']
low_cardinality = ['state']

encoder = TargetEncoder(target_type='continuous', smooth="auto")
X_train[high_cardinality] = encoder.fit_transform(X_train[high_cardinality], y_train)
X_test[high_cardinality] = encoder.transform(X_test[high_cardinality])

# Visualize Encoding
plt.figure(figsize=(12, 6))
sns.kdeplot(X_train['city'], label='Encoded City Values', fill=True)
plt.title('Target Encoding Distribution for Cities', fontsize=14)
plt.xlabel('Encoded Value')
plt.ylabel('Density')
plt.show()

# One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=low_cardinality, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_cardinality, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% 6. Model Development
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

# %% 7. Results Visualization
# Results dataframe
results_df = pd.DataFrame([r for r in results if r]).sort_values('RMSE')

# Horizontal Bar Chart for Model Performance
plt.figure(figsize=(12, 8))
sns.barplot(
    x='RMSE',
    y='Model',
    data=results_df.sort_values('RMSE'),
    palette='viridis'
)
plt.title('Model Performance: RMSE by Model', fontsize=16, pad=15)
plt.xlabel('Root Mean Squared Error (USD)', fontsize=14)
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Residual Analysis
# Train the best model and calculate residuals
best_model = models[results_df.iloc[0]['Model']].fit(X_train_scaled, y_train)
y_pred = np.expm1(best_model.predict(X_test_scaled))
residuals = np.expm1(y_test) - y_pred

# Improved Residual Histogram
plt.figure(figsize=(14, 8))
sns.histplot(
    x=residuals,
    bins=40,  # Reduced bins for a smoother histogram
    kde=True,
    color='darkblue',
    line_kws={'linewidth': 2}  # Thicker KDE line for emphasis
)
plt.title('Residual Distribution of Best Model', fontsize=16, pad=15)
plt.xlabel('Prediction Error (USD)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
plt.text(
    x=residuals.mean(),
    y=plt.ylim()[1] * 0.9,
    s=f"Mean Error: ${residuals.mean():,.0f}",
    color='black',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
# Results Table 
print("Final Model Performance:")

# Ensure numeric types and formatting
results_df['RMSE'] = pd.to_numeric(results_df['RMSE'])
results_df['R2'] = pd.to_numeric(results_df['R2'])

results_df
# %%
