"""
questions_m5.py — Module 5: Credit Default Prediction Lab
Dataset: 10,000 rows x 50 variables, ~23% default rate, seed=505
No demographics — pure credit behaviour variables
M5_1: Logistic Regression (10 steps)
M5_2: XGBoost + SHAP (8 steps)
Format: guided lab — no scoring, correct output shown after each step
"""

import hashlib, re, io, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

def _exec(code, globs):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:    exec(code, globs)
    except Exception as e: sys.stdout = old; return f"ERROR: {e}"
    sys.stdout = old
    return buf.getvalue().strip()

# ════════════════════════════════════════════════════════════════
# DATASET GENERATION (seed=505)
# ════════════════════════════════════════════════════════════════
DF_SETUP = '''\
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.random.seed(505)
n = 10000
avg_utilization      = np.round(np.random.beta(2,5,n),3)
max_utilization      = np.round(np.clip(avg_utilization+np.random.beta(1,4,n)*0.4,0,1),3)
utilization_trend    = np.round(np.random.normal(0,0.1,n),3)
credit_limit         = np.random.randint(10000,500000,n)
limit_increase_count = np.random.randint(0,5,n)
payment_ratio        = np.round(np.random.beta(4,2,n),3)
min_pay_flag         = np.random.choice([0,1],n,p=[0.75,0.25])
months_since_payment = np.random.randint(0,12,n)
avg_monthly_spend    = np.random.randint(1000,80000,n).astype(float)
max_spend_3m         = (avg_monthly_spend*np.random.uniform(1,2.5,n)).astype(int)
spend_trend          = np.round(np.random.normal(0,500,n),0)
num_inquiries        = np.random.randint(0,10,n).astype(float)
num_active_loans     = np.random.randint(0,6,n)
total_outstanding    = np.random.randint(0,2000000,n).astype(float)
months_active        = np.random.randint(3,120,n)
num_products         = np.random.randint(1,6,n)
dpd_30_flag          = np.random.choice([0,1],n,p=[0.80,0.20])
dpd_60_flag          = np.random.choice([0,1],n,p=[0.90,0.10])
times_delinquent     = np.random.randint(0,5,n)
revolving_balance    = np.random.randint(0,300000,n).astype(float)
cash_advance_count   = np.random.randint(0,8,n)
cash_advance_ratio   = np.round(np.random.beta(1,8,n),3)
balance_to_limit     = np.round(avg_utilization*np.random.uniform(0.8,1.1,n),3).clip(0,1)
payment_delay_days   = np.random.randint(0,30,n)
num_late_payments    = np.random.randint(0,6,n)
tenure_months        = months_active.copy()
spend_volatility     = np.round(np.random.exponential(5000,n),0).astype(float)
limit_utilization_3m = np.round(np.random.beta(2,4,n),3)
prev_default_flag    = np.random.choice([0,1],n,p=[0.88,0.12])
account_age_score    = np.round(np.log1p(months_active)/np.log1p(120),3)
payment_consistency  = np.round(np.random.beta(5,2,n),3)
_noise = {f"noise_{i}": np.random.uniform(0,1,n) for i in range(1,16)}
low_var_1   = np.random.choice([0,1],n,p=[0.99,0.01])
low_var_2   = np.zeros(n); low_var_2[:50]=1
high_corr_1 = avg_utilization+np.random.normal(0,0.01,n)
high_corr_2 = credit_limit/1000
log_odds = (-2.5+2.0*avg_utilization+1.5*dpd_30_flag+2.5*dpd_60_flag+1.8*prev_default_flag+1.2*min_pay_flag+0.8*times_delinquent/5-1.0*payment_ratio-0.5*account_age_score+0.5*cash_advance_ratio+np.random.normal(0,0.5,n))
prob   = 1/(1+np.exp(-log_odds))
target = (np.random.uniform(0,1,n)<prob).astype(int)
df = pd.DataFrame({
    "avg_utilization":avg_utilization,"max_utilization":max_utilization,
    "utilization_trend":utilization_trend,"credit_limit":credit_limit,
    "limit_increase_count":limit_increase_count,"payment_ratio":payment_ratio,
    "min_pay_flag":min_pay_flag,"months_since_payment":months_since_payment,
    "avg_monthly_spend":avg_monthly_spend,"max_spend_3m":max_spend_3m,
    "spend_trend":spend_trend,"num_inquiries":num_inquiries,
    "num_active_loans":num_active_loans,"total_outstanding":total_outstanding,
    "months_active":months_active,"num_products":num_products,
    "dpd_30_flag":dpd_30_flag,"dpd_60_flag":dpd_60_flag,
    "times_delinquent":times_delinquent,"revolving_balance":revolving_balance,
    "cash_advance_count":cash_advance_count,"cash_advance_ratio":cash_advance_ratio,
    "balance_to_limit":balance_to_limit,"payment_delay_days":payment_delay_days,
    "num_late_payments":num_late_payments,"tenure_months":tenure_months,
    "spend_volatility":spend_volatility,"limit_utilization_3m":limit_utilization_3m,
    "prev_default_flag":prev_default_flag,"account_age_score":account_age_score,
    "payment_consistency":payment_consistency,
    **_noise,
    "low_var_1":low_var_1,"low_var_2":low_var_2,
    "high_corr_1":high_corr_1,"high_corr_2":high_corr_2,
    "default_next_cycle":target,
})
for _col in ["avg_monthly_spend","total_outstanding","revolving_balance","spend_volatility","num_inquiries"]:
    df.loc[np.random.choice(n,int(n*0.05),replace=False),_col]=np.nan
'''

# Full preprocessing setup (used for steps after step 4)
PREPROC_SETUP = DF_SETUP + '''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: fill missing
for _c in df.select_dtypes(include="number").columns:
    if _c != "default_next_cycle":
        df[_c] = df[_c].fillna(df[_c].median())

# Step 3: outlier clipping
_num = [c for c in df.select_dtypes(include="number").columns if c!="default_next_cycle"]
for _c in _num:
    df[_c] = df[_c].clip(df[_c].quantile(0.05), df[_c].quantile(0.95))

# Step 4: variable reduction
X = df.drop("default_next_cycle", axis=1)
y = df["default_next_cycle"]
_zv = [c for c in X.columns if X[c].var()==0]
X = X.drop(columns=_zv)
_corr = X.corr().abs()
_upper = _corr.where(np.triu(np.ones(_corr.shape),k=1).astype(bool))
_hc = [c for c in _upper.columns if any(_upper[c]>0.85)]
X = X.drop(columns=_hc)

# Step 5: split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
FEATURES  = list(X.columns)
'''

# ════════════════════════════════════════════════════════════════
# MODULE 5_1 — LOGISTIC REGRESSION LAB STEPS
# ════════════════════════════════════════════════════════════════
M5_1_STEPS = [
    {
        "id": 1,
        "title": "Step 1 — Explore the Dataset",
        "context": (
            "You are working on a credit card default prediction problem.\n"
            "The dataset has 10,000 customers and 50 variables — all credit behaviour features.\n"
            "Target: **default_next_cycle** (1 = customer will default in next billing cycle)\n\n"
            "**Task:** Print the following:\n"
            "1. Shape of the dataset\n"
            "2. Default rate (mean of target column, rounded to 3 decimal places)\n"
            "3. Total missing values across all columns"
        ),
        "preload": DF_SETUP,
        "exp": "(10000, 51)\n0.234\n2500",
        "solution": (
            "print(df.shape)\n"
            "print(round(df['default_next_cycle'].mean(), 3))\n"
            "print(df.isnull().sum().sum())"
        ),
        "explanation": (
            "Shape: 10,000 rows × 51 columns (50 features + 1 target).\n"
            "Default rate: ~23.4% — higher than typical real-world datasets (~10-15%), "
            "but useful for demonstrating imbalance handling.\n"
            "Missing: 2,500 total across 5 columns (500 each) — ~5% missingness per affected column."
        ),
    },
    {
        "id": 2,
        "title": "Step 2 — Missing Value Treatment",
        "context": (
            "The dataset has missing values in 5 numeric columns:\n"
            "avg_monthly_spend, total_outstanding, revolving_balance, spend_volatility, num_inquiries\n\n"
            "**Strategy:** For all numeric columns, fill missing values with the **column median**.\n"
            "Median is preferred over mean for financial data — it's robust to outliers.\n\n"
            "**Task:** Fill all numeric columns (except target) with their median.\n"
            "Print total missing values after treatment."
        ),
        "preload": DF_SETUP,
        "exp": "0",
        "solution": (
            "for col in df.select_dtypes(include='number').columns:\n"
            "    if col != 'default_next_cycle':\n"
            "        df[col] = df[col].fillna(df[col].median())\n"
            "print(df.isnull().sum().sum())"
        ),
        "explanation": (
            "After median imputation, all missing values are filled — total = 0.\n"
            "Median is better than mean for skewed distributions like spend and outstanding balances.\n"
            "We exclude the target column 'default_next_cycle' from imputation."
        ),
    },
    {
        "id": 3,
        "title": "Step 3 — Outlier Treatment (Percentile Capping)",
        "context": (
            "Financial variables often have extreme outliers — a customer with ₹50L spend vs typical ₹10K.\n"
            "These can distort model coefficients, especially in Logistic Regression.\n\n"
            "**Strategy:** Winsorize — clip each numeric feature (except target) to its 5th–95th percentile.\n\n"
            "**Task:** Apply percentile clipping to all numeric columns (except default_next_cycle).\n"
            "Print max value of avg_monthly_spend after clipping, rounded to 0 decimal places."
        ),
        "preload": DF_SETUP + "\nfor _c in df.select_dtypes(include='number').columns:\n    if _c!='default_next_cycle': df[_c]=df[_c].fillna(df[_c].median())\n",
        "exp": "75574.0",
        "solution": (
            "num_cols = [c for c in df.select_dtypes(include='number').columns\n"
            "            if c != 'default_next_cycle']\n"
            "for col in num_cols:\n"
            "    df[col] = df[col].clip(df[col].quantile(0.05), df[col].quantile(0.95))\n"
            "print(round(df['avg_monthly_spend'].max(), 0))"
        ),
        "explanation": (
            "Clipping to 5th–95th percentile removes extreme values without dropping rows.\n"
            "avg_monthly_spend max after clipping = 75,574 (was potentially much higher).\n"
            "This is 'winsorization' — standard practice in credit risk preprocessing."
        ),
    },
    {
        "id": 4,
        "title": "Step 4 — Variable Reduction",
        "context": (
            "Our dataset has 50 features, including noise and redundant variables.\n"
            "We need to remove:\n"
            "1. **Zero variance columns** — same value for every row (useless for model)\n"
            "2. **Highly correlated columns** — correlation > 0.85 (redundant information)\n\n"
            "**Task:**\n"
            "1. Separate X (features) and y (target)\n"
            "2. Drop zero-variance columns\n"
            "3. Compute correlation matrix, drop columns with correlation > 0.85 with any other\n"
            "4. Print number of features remaining"
        ),
        "preload": DF_SETUP + "\nfor _c in df.select_dtypes(include='number').columns:\n    if _c!='default_next_cycle': df[_c]=df[_c].fillna(df[_c].median())\n_num=[c for c in df.select_dtypes(include='number').columns if c!='default_next_cycle']\nfor _c in _num: df[_c]=df[_c].clip(df[_c].quantile(0.05),df[_c].quantile(0.95))\n",
        "exp": "41",
        "solution": (
            "X = df.drop('default_next_cycle', axis=1)\n"
            "y = df['default_next_cycle']\n\n"
            "# Drop zero variance\n"
            "zero_var = [c for c in X.columns if X[c].var() == 0]\n"
            "X = X.drop(columns=zero_var)\n\n"
            "# Drop highly correlated\n"
            "corr = X.corr().abs()\n"
            "upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\n"
            "high_corr = [c for c in upper.columns if any(upper[c] > 0.85)]\n"
            "X = X.drop(columns=high_corr)\n\n"
            "print(X.shape[1])"
        ),
        "explanation": (
            "Zero variance columns dropped: low_var_1, low_var_2 (near-constant values).\n"
            "High correlation (>0.85) dropped: max_utilization, max_spend_3m, balance_to_limit, "
            "tenure_months, account_age_score, high_corr_1, high_corr_2.\n"
            "Result: 50 → 41 features. Correlation matrix upper triangle avoids double-counting."
        ),
    },
    {
        "id": 5,
        "title": "Step 5 — Train/Test Split (Stratified)",
        "context": (
            "We split data 80/20 into training and test sets.\n"
            "**Stratified split** is critical — it preserves the 23% default rate in both sets.\n"
            "Without stratification, one set might get fewer defaulters by chance.\n\n"
            "**Task:** Split X and y with:\n"
            "- test_size=0.2, random_state=42, stratify=y\n\n"
            "Print:\n"
            "1. Training set shape\n"
            "2. Test set shape\n"
            "3. Default rate in training set (rounded to 3 decimal places)"
        ),
        "preload": PREPROC_SETUP.split("# Step 5")[0],
        "exp": "(8000, 41)\n(2000, 41)\n0.234",
        "solution": (
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
            "print(X_train.shape)\n"
            "print(X_test.shape)\n"
            "print(round(y_train.mean(), 3))"
        ),
        "explanation": (
            "Train: 8,000 rows, Test: 2,000 rows.\n"
            "Default rate preserved at 0.234 in training set — same as full dataset.\n"
            "Stratified split is mandatory for imbalanced classification problems."
        ),
    },
    {
        "id": 6,
        "title": "Step 6 — Feature Scaling",
        "context": (
            "Logistic Regression is sensitive to feature scale.\n"
            "credit_limit ranges in hundreds of thousands while payment_ratio is 0-1.\n"
            "Without scaling, large-magnitude features dominate the coefficients.\n\n"
            "**Strategy:** StandardScaler — transforms each feature to mean=0, std=1.\n"
            "**Important:** Fit scaler on training data only, then transform both train and test.\n"
            "Never fit on test data — that would cause data leakage.\n\n"
            "**Task:** Scale X_train and X_test. Print mean of first column of X_train_scaled (rounded to 2 dp)."
        ),
        "preload": PREPROC_SETUP.split("# Step 6")[0],
        "exp": "-0.0",
        "solution": (
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "X_test_s  = scaler.transform(X_test)\n"
            "print(round(X_train_s[:, 0].mean(), 2))"
        ),
        "explanation": (
            "Mean of scaled training column = ~0.0 (exactly 0 by design of StandardScaler).\n"
            "fit_transform on train: learns mean/std from training data.\n"
            "transform on test: applies same mean/std — no information from test leaks into scaler."
        ),
    },
    {
        "id": 7,
        "title": "Step 7 — Train Logistic Regression",
        "context": (
            "Now we train the model.\n"
            "We use class_weight='balanced' to handle class imbalance (23% defaulters vs 77% non-defaulters).\n"
            "Without this, the model would bias toward predicting 0 (majority class) always.\n\n"
            "**Task:** Train LogisticRegression with:\n"
            "- max_iter=1000, random_state=42, class_weight='balanced'\n\n"
            "Print AUC-ROC on test set (rounded to 4 decimal places).\n"
            "AUC > 0.7 is a good starting point for credit risk models."
        ),
        "preload": PREPROC_SETUP,
        "exp": "0.7961",
        "solution": (
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import roc_auc_score\n\n"
            "lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
            "lr.fit(X_train_s, y_train)\n"
            "lr_pred = lr.predict_proba(X_test_s)[:, 1]\n"
            "print(round(roc_auc_score(y_test, lr_pred), 4))"
        ),
        "explanation": (
            "AUC-ROC = 0.7961. This means the model correctly ranks a random defaulter "
            "above a random non-defaulter ~80% of the time.\n"
            "In credit risk, AUC > 0.75 is considered acceptable. > 0.80 is good.\n"
            "class_weight='balanced' automatically adjusts weights inversely proportional to class frequency."
        ),
    },
    {
        "id": 8,
        "title": "Step 8 — Classification Report",
        "context": (
            "AUC measures ranking ability. The classification report shows precision, recall and F1 "
            "at the default threshold of 0.5.\n\n"
            "**Key metrics for collections:**\n"
            "- **Recall (defaulters)**: How many actual defaulters did we catch?\n"
            "- **Precision (defaulters)**: Of those we flagged, how many actually defaulted?\n\n"
            "**Task:** Predict class labels using threshold 0.5. Print the classification report."
        ),
        "preload": PREPROC_SETUP + "\nfrom sklearn.linear_model import LogisticRegression\nlr=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')\nlr.fit(X_train_s,y_train)\nlr_pred=lr.predict_proba(X_test_s)[:,1]\n",
        "exp": "              precision    recall  f1-score   support\n\n           0       0.88      0.74      0.81      1531\n           1       0.45      0.68      0.54       469\n\n    accuracy                           0.73      2000\n   macro avg       0.67      0.71      0.67      2000\nweighted avg       0.78      0.73      0.74      2000",
        "solution": (
            "from sklearn.metrics import classification_report\n"
            "lr_class = lr.predict(X_test_s)\n"
            "print(classification_report(y_test, lr_class, digits=2))"
        ),
        "explanation": (
            "Recall for defaulters (class 1) = 0.68 — we catch 68% of actual defaulters.\n"
            "Precision for defaulters = 0.45 — of those we flag, 45% actually default.\n"
            "In collections, high recall is usually preferred over high precision — "
            "missing a defaulter is more costly than a false alarm."
        ),
    },
    {
        "id": 9,
        "title": "Step 9 — Confusion Matrix",
        "context": (
            "The confusion matrix breaks down predictions into 4 categories:\n"
            "- **True Negative (TN)**: Correctly predicted non-defaulter\n"
            "- **False Positive (FP)**: Predicted default, actually didn't\n"
            "- **False Negative (FN)**: Missed a defaulter ← most costly in collections\n"
            "- **True Positive (TP)**: Correctly caught a defaulter\n\n"
            "**Task:** Print the confusion matrix at threshold 0.5."
        ),
        "preload": PREPROC_SETUP + "\nfrom sklearn.linear_model import LogisticRegression\nlr=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')\nlr.fit(X_train_s,y_train)\nlr_class=lr.predict(X_test_s)\n",
        "exp": "[[1135  396]\n [ 148  321]]",
        "solution": (
            "from sklearn.metrics import confusion_matrix\n"
            "print(confusion_matrix(y_test, lr_class))"
        ),
        "explanation": (
            "TN=1135, FP=396, FN=148, TP=321.\n"
            "False Negatives = 148: these are defaulters we MISSED — high business cost.\n"
            "False Positives = 396: non-defaulters we flagged — lower cost, just unnecessary contact.\n"
            "Lowering the threshold captures more defaulters (reduces FN) but increases FP."
        ),
    },
    {
        "id": 10,
        "title": "Step 10 — Threshold Tuning",
        "context": (
            "The default threshold of 0.5 is rarely optimal for imbalanced datasets.\n"
            "Lowering the threshold catches more defaulters (higher recall) at the cost of more false alarms.\n\n"
            "**Task:** Test thresholds [0.3, 0.4, 0.5, 0.6].\n"
            "For each, compute F1 score for class 1 (defaulters).\n"
            "Print the best threshold and its F1 score (rounded to 4 decimal places), space-separated."
        ),
        "preload": PREPROC_SETUP + "\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import f1_score\nlr=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')\nlr.fit(X_train_s,y_train)\nlr_pred=lr.predict_proba(X_test_s)[:,1]\n",
        "exp": "0.6 0.5445",
        "solution": (
            "from sklearn.metrics import f1_score\n"
            "best_t, best_f1 = 0.5, 0\n"
            "for t in [0.3, 0.4, 0.5, 0.6]:\n"
            "    preds = (lr_pred >= t).astype(int)\n"
            "    f1 = f1_score(y_test, preds)\n"
            "    if f1 > best_f1:\n"
            "        best_f1 = f1; best_t = t\n"
            "print(best_t, round(best_f1, 4))"
        ),
        "explanation": (
            "Best threshold = 0.6 with F1 = 0.5445.\n"
            "In real credit models, the threshold is set based on business rules:\n"
            "- Collections capacity: how many customers can your team call?\n"
            "- Cost of default vs cost of false alarm\n"
            "F1 balances precision and recall — useful when both matter."
        ),
    },
]

# ════════════════════════════════════════════════════════════════
# MODULE 5_2 — XGBOOST + SHAP LAB STEPS
# ════════════════════════════════════════════════════════════════
M5_2_STEPS = [
    {
        "id": 1,
        "title": "Step 1 — Load Preprocessed Data",
        "context": (
            "In this lab we use the same dataset as Module 5_1, with the same preprocessing pipeline:\n"
            "✅ Missing values filled (median)\n"
            "✅ Outliers clipped (5th–95th percentile)\n"
            "✅ Variable reduction (zero variance + high correlation dropped)\n"
            "✅ Train/test split (80/20, stratified, random_state=42)\n"
            "✅ StandardScaler applied\n\n"
            "**Task:** Confirm the preprocessing is correct.\n"
            "Print: number of features, training set size, test default rate (rounded to 3 dp)."
        ),
        "preload": PREPROC_SETUP,
        "exp": "41\n8000\n0.234",
        "solution": (
            "print(len(FEATURES))\n"
            "print(X_train_s.shape[0])\n"
            "print(round(y_test.mean(), 3))"
        ),
        "explanation": (
            "41 features after reduction. 8,000 training rows. Test default rate = 0.234 — "
            "same as full dataset, confirming stratified split worked correctly."
        ),
    },
    {
        "id": 2,
        "title": "Step 2 — Train XGBoost (Default Params)",
        "context": (
            "XGBoost is a gradient boosting algorithm — it builds trees sequentially, "
            "each correcting errors of the previous.\n\n"
            "We use scale_pos_weight=3 to handle class imbalance "
            "(ratio of negatives to positives ≈ 3:1).\n\n"
            "**Task:** Train XGBClassifier with:\n"
            "- n_estimators=100, max_depth=4, learning_rate=0.1\n"
            "- scale_pos_weight=3, random_state=42, eval_metric='logloss', verbosity=0\n\n"
            "Print AUC-ROC on test set (rounded to 4 decimal places)."
        ),
        "preload": PREPROC_SETUP,
        "exp": "0.7877",
        "solution": (
            "from xgboost import XGBClassifier\n"
            "from sklearn.metrics import roc_auc_score\n\n"
            "xgb = XGBClassifier(\n"
            "    n_estimators=100, max_depth=4, learning_rate=0.1,\n"
            "    scale_pos_weight=3, random_state=42,\n"
            "    eval_metric='logloss', verbosity=0\n"
            ")\n"
            "xgb.fit(X_train_s, y_train)\n"
            "xgb_pred = xgb.predict_proba(X_test_s)[:, 1]\n"
            "print(round(roc_auc_score(y_test, xgb_pred), 4))"
        ),
        "explanation": (
            "XGBoost AUC = 0.7877 vs Logistic Regression AUC = 0.7961.\n"
            "LR slightly outperforms XGBoost here — this happens with well-structured tabular data "
            "where linear relationships dominate.\n"
            "XGBoost shines when feature interactions and non-linearity are present."
        ),
    },
    {
        "id": 3,
        "title": "Step 3 — Feature Importance",
        "context": (
            "XGBoost computes feature importance as the number of times each feature is used "
            "to split data across all trees.\n\n"
            "**Task:** Extract feature importances from the trained XGBoost model.\n"
            "Print the top 10 features and their importance scores as a dict (rounded to 4 dp)."
        ),
        "preload": PREPROC_SETUP + "\nfrom xgboost import XGBClassifier\nxgb=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,scale_pos_weight=3,random_state=42,eval_metric='logloss',verbosity=0)\nxgb.fit(X_train_s,y_train)\n",
        "exp": "{'dpd_60_flag': 0.1774, 'dpd_30_flag': 0.0969, 'prev_default_flag': 0.08, 'min_pay_flag': 0.0684, 'avg_utilization': 0.024, 'times_delinquent': 0.0216, 'payment_ratio': 0.0182, 'num_inquiries': 0.0182, 'noise_4': 0.0171, 'months_active': 0.017}",
        "solution": (
            "import pandas as pd\n"
            "fi = pd.Series(xgb.feature_importances_, index=FEATURES)\n"
            "fi = fi.sort_values(ascending=False)\n"
            "print(fi.head(10).round(4).to_dict())"
        ),
        "explanation": (
            "Top features: dpd_60_flag, dpd_30_flag, prev_default_flag, min_pay_flag, avg_utilization.\n"
            "These align with credit risk domain knowledge — delinquency history and payment behaviour "
            "are the strongest default predictors.\n"
            "Note: noise_4 appears in top 10 — XGBoost can overfit to noise variables. "
            "SHAP values (next steps) give a more reliable importance measure."
        ),
    },
    {
        "id": 4,
        "title": "Step 4 — Cross-Validation (Stratified K-Fold)",
        "context": (
            "A single train/test split can be lucky or unlucky depending on which customers end up in each set.\n"
            "Cross-validation gives a more reliable estimate of model performance.\n\n"
            "**StratifiedKFold** preserves the class ratio in each fold.\n\n"
            "**Task:** Run 5-fold stratified cross-validation on the training set.\n"
            "Use the same XGBoost params. Scoring = 'roc_auc'.\n"
            "Print mean AUC and std (both rounded to 4 decimal places), space-separated."
        ),
        "preload": PREPROC_SETUP,
        "exp": "0.7741 0.0104",
        "solution": (
            "from xgboost import XGBClassifier\n"
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n\n"
            "xgb = XGBClassifier(\n"
            "    n_estimators=100, max_depth=4, learning_rate=0.1,\n"
            "    scale_pos_weight=3, random_state=42,\n"
            "    eval_metric='logloss', verbosity=0\n"
            ")\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "scores = cross_val_score(xgb, X_train_s, y_train, cv=cv, scoring='roc_auc')\n"
            "print(round(scores.mean(), 4), round(scores.std(), 4))"
        ),
        "explanation": (
            "CV AUC = 0.7741 ± 0.0104. Low std means the model is stable across folds.\n"
            "CV AUC (0.7741) is slightly lower than test AUC (0.7877) — this is normal.\n"
            "Always report CV performance in production — it's more honest than a single split."
        ),
    },
    {
        "id": 5,
        "title": "Step 5 — Hyperparameter Tuning",
        "context": (
            "We test 3 XGBoost configurations and pick the one with best test AUC.\n\n"
            "**Configurations to test:**\n"
            "1. n_estimators=100, max_depth=4, learning_rate=0.1\n"
            "2. n_estimators=200, max_depth=5, learning_rate=0.05\n"
            "3. n_estimators=150, max_depth=6, learning_rate=0.05\n"
            "All with scale_pos_weight=3, random_state=42, eval_metric='logloss', verbosity=0\n\n"
            "**Task:** Print best config number (1,2,3) and its AUC (rounded to 4 dp), space-separated."
        ),
        "preload": PREPROC_SETUP,
        "exp": "1 0.7877",
        "solution": (
            "from xgboost import XGBClassifier\n"
            "from sklearn.metrics import roc_auc_score\n\n"
            "configs = [\n"
            "    {'n_estimators':100,'max_depth':4,'learning_rate':0.1},\n"
            "    {'n_estimators':200,'max_depth':5,'learning_rate':0.05},\n"
            "    {'n_estimators':150,'max_depth':6,'learning_rate':0.05},\n"
            "]\n"
            "best_i, best_auc = 1, 0\n"
            "for i, cfg in enumerate(configs, 1):\n"
            "    xgb = XGBClassifier(**cfg, scale_pos_weight=3, random_state=42,\n"
            "                        eval_metric='logloss', verbosity=0)\n"
            "    xgb.fit(X_train_s, y_train)\n"
            "    auc = round(roc_auc_score(y_test, xgb.predict_proba(X_test_s)[:,1]), 4)\n"
            "    if auc > best_auc: best_auc=auc; best_i=i\n"
            "print(best_i, best_auc)"
        ),
        "explanation": (
            "Config 1 (n=100, depth=4, lr=0.1) gives best AUC = 0.7877.\n"
            "Deeper trees and more estimators don't always help — they can overfit.\n"
            "In practice, use GridSearchCV or Optuna for systematic hyperparameter search."
        ),
    },
    {
        "id": 6,
        "title": "Step 6 — SHAP Values (Model Explainability)",
        "context": (
            "Feature importance from XGBoost tells you which features are used most.\n"
            "SHAP (SHapley Additive exPlanations) tells you **how much each feature pushes "
            "the prediction up or down for each customer**.\n\n"
            "SHAP is the gold standard for credit risk model explainability — "
            "regulators often require it.\n\n"
            "**Task:** Compute SHAP values on first 500 test samples.\n"
            "Print the top 10 features by mean absolute SHAP value as a dict (rounded to 4 dp)."
        ),
        "preload": PREPROC_SETUP + "\nfrom xgboost import XGBClassifier\nxgb=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,scale_pos_weight=3,random_state=42,eval_metric='logloss',verbosity=0)\nxgb.fit(X_train_s,y_train)\n",
        "exp": "{'dpd_30_flag': 0.4401, 'dpd_60_flag': 0.4361, 'min_pay_flag': 0.3813, 'prev_default_flag': 0.2862, 'avg_utilization': 0.2199, 'times_delinquent': 0.1891, 'payment_ratio': 0.1248, 'months_active': 0.0707, 'spend_volatility': 0.0578, 'noise_4': 0.0531}",
        "solution": (
            "import shap, pandas as pd, numpy as np\n\n"
            "explainer   = shap.TreeExplainer(xgb)\n"
            "shap_values = explainer.shap_values(X_test_s[:500])\n"
            "shap_imp    = pd.Series(\n"
            "    np.abs(shap_values).mean(axis=0),\n"
            "    index=FEATURES\n"
            ").sort_values(ascending=False)\n"
            "print(shap_imp.head(10).round(4).to_dict())"
        ),
        "explanation": (
            "SHAP top features: dpd_30_flag, dpd_60_flag, min_pay_flag, prev_default_flag, avg_utilization.\n"
            "These align with domain knowledge — delinquency history is the strongest signal.\n"
            "SHAP is more reliable than feature importance: it accounts for feature interactions "
            "and gives directional impact (positive = increases default probability).\n"
            "Notice noise_4 drops significantly compared to feature importance — SHAP filters noise better."
        ),
    },
    {
        "id": 7,
        "title": "Step 7 — LR vs XGBoost Comparison",
        "context": (
            "Now we compare both models head-to-head on the same test set.\n\n"
            "**Task:** Train both models and print AUC for each:\n"
            "- LogisticRegression: max_iter=1000, random_state=42, class_weight='balanced'\n"
            "- XGBClassifier: n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=3, random_state=42\n\n"
            "Print:\n"
            "LR AUC: X.XXXX\n"
            "XGB AUC: X.XXXX\n"
            "Winner: [LR/XGBoost]"
        ),
        "preload": PREPROC_SETUP,
        "exp": "LR AUC: 0.7961\nXGB AUC: 0.7877\nWinner: LR",
        "solution": (
            "from sklearn.linear_model import LogisticRegression\n"
            "from xgboost import XGBClassifier\n"
            "from sklearn.metrics import roc_auc_score\n\n"
            "lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
            "lr.fit(X_train_s, y_train)\n"
            "lr_auc = round(roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1]), 4)\n\n"
            "xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,\n"
            "                    scale_pos_weight=3, random_state=42,\n"
            "                    eval_metric='logloss', verbosity=0)\n"
            "xgb.fit(X_train_s, y_train)\n"
            "xgb_auc = round(roc_auc_score(y_test, xgb.predict_proba(X_test_s)[:,1]), 4)\n\n"
            "print(f'LR AUC: {lr_auc}')\n"
            "print(f'XGB AUC: {xgb_auc}')\n"
            "print('Winner:', 'LR' if lr_auc > xgb_auc else 'XGBoost')"
        ),
        "explanation": (
            "LR wins (0.7961 vs 0.7877) on this dataset.\n"
            "This is common in credit risk: data is often well-structured with strong linear signals "
            "(delinquency flags, utilization ratios) where LR excels.\n"
            "XGBoost typically outperforms on complex datasets with many feature interactions.\n"
            "Always compare multiple models — never assume tree-based = better."
        ),
    },
    {
        "id": 8,
        "title": "Step 8 — Business Threshold Analysis",
        "context": (
            "In real collections, we don't just predict — we set a risk threshold.\n"
            "Customers above the threshold get a collections call.\n\n"
            "**Trade-off:**\n"
            "- Lower threshold → catch more defaulters (higher recall) → more calls, higher cost\n"
            "- Higher threshold → fewer but more accurate alerts (higher precision) → fewer calls\n\n"
            "**Task:** For XGBoost predictions at thresholds [0.3, 0.4, 0.5]:\n"
            "Print precision, recall and F1 for each threshold in format:\n"
            "t=X.X: P=X.XXX R=X.XXX F1=X.XXX"
        ),
        "preload": PREPROC_SETUP + "\nfrom xgboost import XGBClassifier\nxgb=XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,scale_pos_weight=3,random_state=42,eval_metric='logloss',verbosity=0)\nxgb.fit(X_train_s,y_train)\nxgb_pred=xgb.predict_proba(X_test_s)[:,1]\n",
        "exp": "t=0.3: P=0.361 R=0.849 F1=0.507\nt=0.4: P=0.412 R=0.725 F1=0.526\nt=0.5: P=0.476 R=0.569 F1=0.518",
        "solution": (
            "from sklearn.metrics import precision_score, recall_score, f1_score\n\n"
            "for t in [0.3, 0.4, 0.5]:\n"
            "    preds = (xgb_pred >= t).astype(int)\n"
            "    p = round(precision_score(y_test, preds), 3)\n"
            "    r = round(recall_score(y_test, preds), 3)\n"
            "    f = round(f1_score(y_test, preds), 3)\n"
            "    print(f't={t}: P={p} R={r} F1={f}')"
        ),
        "explanation": (
            "t=0.3: High recall (0.849) — catches 85% of defaulters but low precision (0.361) — many false alarms.\n"
            "t=0.5: Higher precision (0.476) but lower recall (0.569) — misses 43% of defaulters.\n"
            "t=0.4: Best F1 (0.526) — balanced trade-off.\n\n"
            "Business decision: If collections team can call 500 customers, use t=0.4.\n"
            "If capacity is 200, use t=0.5 to target only highest-risk customers."
        ),
    },
]

# ── MAPS ──────────────────────────────────────────────────────────
M5_1_LAB = M5_1_STEPS
M5_2_LAB = M5_2_STEPS
