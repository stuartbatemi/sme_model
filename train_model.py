"""
train_model.py
===============
BEGINNER EXPLANATION:
This trains your TWO-STAGE model:

  STAGE 1 (Multiple Linear Regression):
    Input  -> business features (location, capital, sector, workers, etc.)
    Output -> a number: "Expected Monthly Profit (TZS)"

  STAGE 2 (Random Forest + XGBoost classifiers):
    Input  -> the same features PLUS Stage 1's predicted profit
    Output -> a category: "High" / "Medium" / "Low" chance of success

  "Success category" is defined using ROI (Return on Investment):
    ROI = (predicted annual profit / startup capital) x 100
    - High   = top third of ROI values
    - Medium = middle third
    - Low    = bottom third

WHAT GETS SAVED (to /mnt/user-data/outputs/models/):
  - stage1_linear_regression.joblib   (the MLR model)
  - stage2_random_forest.joblib       (RF classifier)
  - stage2_xgboost.joblib             (XGBoost classifier)
  - feature_columns.json              (list & order of feature columns)
  - location_lookup.csv               (per-ward/district census + saturation info)
  - activity_catalog.csv              (list of all business activities + their stats)

Run: python3 train_model.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from xgboost import XGBClassifier

from features import load_data, build_features, get_profit_target, SECTOR_NAMES

OUT_DIR = './models'
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load + engineer features
# ------------------------------------------------------------------
print("Loading data...")
df = load_data()
df = build_features(df)
df['Monthly_Profit_TZS'] = get_profit_target(df)

# Fill ward-level saturation count with district-level if ward unknown
df['Ward_Sector_Count'] = df['Ward_Sector_Count'].fillna(df['District_Sector_Count'])
df['Census_WorkingAge_Pop'] = df['Census_WorkingAge_Pop'].fillna(df['Census_WorkingAge_Pop'].median())
df['Ward_Sector_Count'] = df['Ward_Sector_Count'].fillna(0)

# ------------------------------------------------------------------
# 2. Build the feature matrix (X)
# ------------------------------------------------------------------
# BEGINNER: One-hot encoding turns a category like ISIC_Section='G'
# into a set of 0/1 columns, e.g. "Is_Sector_G = 1, Is_Sector_C = 0, ..."
# Models can only do math with numbers, so this is required.

categorical_cols = ['ISIC_Section', 'District_Name']
numeric_cols = ['Log_Capital', 'NumberTotalWorkers', 'Census_WorkingAge_Pop',
                 'Census_Total', 'Ward_Sector_Count']

X = pd.get_dummies(df[categorical_cols + numeric_cols], columns=categorical_cols)
feature_columns = X.columns.tolist()

y_profit = np.log(df['Monthly_Profit_TZS'].clip(lower=1))  # log scale for stability

# ------------------------------------------------------------------
# 3. STAGE 1: Multiple Linear Regression -> predict log(monthly profit)
# ------------------------------------------------------------------
print("\n--- STAGE 1: Linear Regression ---")
X_train, X_test, y_train, y_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)

stage1 = LinearRegression()
stage1.fit(X_train, y_train)

pred_test = stage1.predict(X_test)
print(f"R^2 score: {r2_score(y_test, pred_test):.3f}")
print(f"MAE (log scale): {mean_absolute_error(y_test, pred_test):.3f}")
print("NOTE: The target is now a SOURCED ESTIMATE (anchored to the 2010 MSME")
print("      national survey + NBS sector capital benchmarks), not pure")
print("      placeholder - but it is still a STATISTICAL ESTIMATE, not real")
print("      observed profit. Replace with TRA/survey data when available.")

# Stage 1 prediction becomes an extra feature for Stage 2
df['Stage1_Predicted_LogProfit'] = stage1.predict(X)
X_stage2 = X.copy()
X_stage2['Stage1_Predicted_LogProfit'] = df['Stage1_Predicted_LogProfit']

# ------------------------------------------------------------------
# 4. Build the Success Category target (High / Medium / Low) from ROI
# ------------------------------------------------------------------
predicted_annual_profit = np.exp(df['Stage1_Predicted_LogProfit']) * 12
roi = predicted_annual_profit / df['Estimated_Startup_Capital_TZS']

# Split into 3 equal-sized groups (terciles)
df['ROI'] = roi
labels = pd.qcut(roi, q=3, labels=['Low', 'Medium', 'High'])
df['Success_Category'] = labels

print("\nSuccess_Category distribution:")
print(df['Success_Category'].value_counts())

# ------------------------------------------------------------------
# 5. STAGE 2: Random Forest + XGBoost classifiers
# ------------------------------------------------------------------
print("\n--- STAGE 2: Random Forest ---")
y_class = df['Success_Category']
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_stage2, y_class, test_size=0.2, random_state=42, stratify=y_class)

rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X2_train, y2_train)
rf_pred = rf.predict(X2_test)
print(f"Accuracy: {accuracy_score(y2_test, rf_pred):.3f}")
print(classification_report(y2_test, rf_pred))

print("\n--- STAGE 2: XGBoost ---")
# XGBoost needs numeric labels, not text
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
inv_label_map = {v: k for k, v in label_map.items()}
y2_train_num = y2_train.map(label_map)
y2_test_num = y2_test.map(label_map)

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     random_state=42, eval_metric='mlogloss')
xgb.fit(X2_train, y2_train_num)
xgb_pred = xgb.predict(X2_test)
print(f"Accuracy: {accuracy_score(y2_test_num, xgb_pred):.3f}")
print(classification_report(y2_test_num, xgb_pred, target_names=['Low', 'Medium', 'High']))

# ------------------------------------------------------------------
# 6. Save everything needed for prediction later
# ------------------------------------------------------------------
joblib.dump(stage1, f'{OUT_DIR}/stage1_linear_regression.joblib')
joblib.dump(rf, f'{OUT_DIR}/stage2_random_forest.joblib')
joblib.dump(xgb, f'{OUT_DIR}/stage2_xgboost.joblib')

with open(f'{OUT_DIR}/feature_columns.json', 'w') as f:
    json.dump({'stage1_cols': feature_columns,
                'stage2_cols': X_stage2.columns.tolist(),
                'label_map': label_map}, f, indent=2)

# Location lookup: one row per Ward (or District if ward unknown), with
# census numbers + how many businesses of each sector already exist there
loc_cols = ['Region_Name', 'District_Name', 'Ward_Name', 'Census_WorkingAge_Pop', 'Census_Total']
location_lookup = df[loc_cols].drop_duplicates()
location_lookup.to_csv(f'{OUT_DIR}/location_lookup.csv', index=False)

# Saturation lookup: (District, Ward, ISIC_Section) -> count of existing businesses
saturation_lookup = df.groupby(['District_Name', 'Ward_Name', 'ISIC_Section'], dropna=False) \
                       .size().reset_index(name='Existing_Business_Count')
saturation_lookup.to_csv(f'{OUT_DIR}/saturation_lookup.csv', index=False)

# Activity catalog: every distinct business activity + its sector + typical capital
activity_catalog = df.groupby(['ISIC_Detailed', 'MainActivityDescription', 'ISIC_Section']).agg(
    Typical_Capital_TZS=('Estimated_Startup_Capital_TZS', 'median'),
    Count=('Estimated_Startup_Capital_TZS', 'count')
).reset_index()
activity_catalog['Sector_Name'] = activity_catalog['ISIC_Section'].map(SECTOR_NAMES)
activity_catalog.to_csv(f'{OUT_DIR}/activity_catalog.csv', index=False)

print(f"\nAll model artifacts saved to {OUT_DIR}/")
print("Files:", os.listdir(OUT_DIR))
