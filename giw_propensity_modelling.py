import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# For logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

# For XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# For SHp
import shap

# Import dataset
data_types = {
    'contact_id': 'object'
}

df_giw = pd.read_csv("df_giw.csv", dtype=data_types, index_col=False)

# Fill NAs with 0 and Map "Confirmed" to 1
df_giw['bequest_status'] = df_giw['bequest_status'].fillna(0).replace('Confirmed', 1)

print(df_giw.head())
print(df_giw.info())

# Select necessary columns only
model_df = df_giw[['age_group', 'frequency', 'monetary_value', 'recency', 'tenure', 'rg_status', 'bequest_status']]

# One-hot encode (drop first to avoid dummy trap) - Age
age_dummies = pd.get_dummies(df_giw['age_group'], prefix = "age", drop_first = True)
model_df = pd.concat([model_df, age_dummies], axis = 1)

# One-hot encode (drop first to avoid dummy trap) - RG Status
rg_dummies = pd.get_dummies(model_df['rg_status'], prefix='rg', drop_first=True)

model_df = pd.concat([model_df, rg_dummies], axis=1)

print(model_df.head())
print(model_df.info())

# Drop unnecessary columns
model_df = model_df.drop(columns=['age_group', 'rg_status'])

print(model_df.head())

# Define X, y
X = model_df.drop(columns=['bequest_status'])
y = model_df['bequest_status']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    random_state = 42)

# ---Logistic Regression---

# Scale features
# Identify numeric columns
numeric_cols = X.select_dtypes(include = ["int64", "float64"]).columns # There is no float64

# Identify the remaining columns (assuming they're one-hot encoded or categorical dummies)
non_numeric_cols = X.columns.difference(numeric_cols)

# Initialise the StandardScaler
scaler = StandardScaler()

# Scale only numeric columns
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), 
                           columns = numeric_cols, index = X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_cols]), 
                          columns = numeric_cols, index = X_test.index)

# Keep non-numeric (dummy) columns as-is
X_train_cat = X_train[non_numeric_cols]
X_test_cat = X_test[non_numeric_cols]

# Combine back
X_train_scaled = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_scaled = pd.concat([X_test_num, X_test_cat], axis=1)

# Fit a logistic regression model
clf = LogisticRegression(random_state = 42)
clf.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]  # For ROC-AUC

print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

RocCurveDisplay.from_predictions(y_test, y_proba)

# --- XGBoost ---

# Estimate class imbalance ratio
neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos

print(scale_pos_weight)
# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(
    n_estimators = 100, # Number of boosting rounds
    learning_rate = 0.1, # Boosting learning rate
    max_depth = 3,
    scale_pos_weight = scale_pos_weight,
    random_state = 42)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Evaluate the model
y_pred_xgb = xg_cl.predict(X_test)
y_proba_xgb = xg_cl.predict_proba(X_test)[:, 1]

print("AUC:", roc_auc_score(y_test, y_proba_xgb))
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

RocCurveDisplay.from_predictions(y_test, y_proba_xgb)

# --- SHAP for XGBoost interpretation ---

# Convert bool to float
X_train[non_numeric_cols] = X_train[non_numeric_cols].astype("float")
X_test[non_numeric_cols] = X_test[non_numeric_cols].astype("float")

shap.initjs() # Initialise the necessary JavaScript libraries for rendering interactive SHAP plots

explainer = shap.Explainer(xg_cl, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)