# =====================================
# IMPORT LIBRARIES
# =====================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("C:/Users/admin/Documents/Shreya/ML/lending_club_loan_two.csv")

print("\nDataset Description:")
print(df.describe())

# =====================================
# DATA CLEANING
# =====================================

# Drop duplicates
# df = df.drop_duplicates()

# # Check missing values
# print("\nMissing values before filling:")
# print(df.isnull().sum().sort_values(ascending=False))

# # Fill missing values
df['mort_acc'] = df.groupby('total_acc')['mort_acc'].transform(
    lambda x: x.fillna(x.median())
)
df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].median())
df['emp_title'] = df['emp_title'].fillna('Unknown')
df['emp_length'] = df['emp_length'].fillna('Unknown')
df.drop('title', axis=1, inplace=True)
df['issue_d'] = pd.to_datetime(df['issue_d'])
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

df['credit_age'] = (
    df['issue_d'].dt.year - df['earliest_cr_line'].dt.year
)
df.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=[
    'purpose', 'home_ownership', 'application_type'
], drop_first=True)
df['annual_inc'] = df['annual_inc'].clip(
    upper=df['annual_inc'].quantile(0.99)
)

df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 1,
    'Charged Off': 0
})

df['term'] = df['term'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(0)
df.drop('grade', axis=1, inplace=True)
df['sub_grade'] = df['sub_grade'].astype('category').cat.codes
df['verification_status'] = df['verification_status'].map({
    'Verified': 1,
    'Source Verified': 1,
    'Not Verified': 0
})

df['initial_list_status'] = df['initial_list_status'].map({
    'w': 1,
    'f': 0
})
df.drop('address', axis=1, inplace=True)
df['annual_inc'] = df['annual_inc'].clip(
    upper=df['annual_inc'].quantile(0.99)
)

df['revol_util'] = df['revol_util'].clip(0, 100)
# print(df.isnull().sum().sum()  ) # 0
# print(df.select_dtypes('object'))  # should be empty
# print(df.info())
df.drop('emp_title', axis=1, inplace=True)

emp_length_map = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10,
    'Unknown': 0
}

df['emp_length'] = df['emp_length'].map(emp_length_map)
print(df.select_dtypes(include='object').columns)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

log_model.fit(X_train_scaled, y_train)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)[:, 1]
import numpy as np

threshold = 0.45
y_pred_custom = (y_prob >= threshold).astype(int)

print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_custom))
# =========================================================
# RANDOM FOREST
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print("\nRANDOM FOREST RESULTS")
print("ROC-AUC:", rf_auc)
print(confusion_matrix(y_test, rf_model.predict(X_test)))
print(classification_report(y_test, rf_model.predict(X_test)))

