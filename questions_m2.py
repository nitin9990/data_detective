"""
questions_m2.py — Module 2: Data Preprocessing & Feature Engineering
3 Practice sets + 1 Final assessment (moderate level, 45 min, 1 attempt)
Topics: Missing values, outlier detection, datatype conversion,
        creating new variables, encoding, scaling/normalization
Dataset: Banking loan dataset (seed=202, 300 rows) with intentional missing values & outliers
"""

import hashlib, re, io, sys
import numpy as np
import pandas as pd

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

def _exec(code, globs):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:    exec(code, globs)
    except Exception as e: sys.stdout = old; return f"ERROR: {e}"
    sys.stdout = old
    return buf.getvalue().strip()

# ── DATASET (seed=202, 300 rows) ─────────────────────────────────
np.random.seed(202)
_n = 300

_age = np.random.randint(22, 65, _n).astype(float)
_age[np.random.choice(_n, 15, replace=False)] = np.nan

_income = np.random.randint(200000, 2500000, _n).astype(float)
_income[np.random.choice(_n, 12, replace=False)] = np.nan
_income[np.random.choice(_n, 5, replace=False)] = np.random.randint(8000000, 12000000, 5)

_cs = np.random.randint(500, 850, _n).astype(float)
_cs[np.random.choice(_n, 10, replace=False)] = np.nan

_la = np.random.randint(100000, 2000000, _n).astype(float)
_la[np.random.choice(_n, 8, replace=False)] = np.nan

_emp = np.random.choice(['Salaried','Self-Employed','Business','Unemployed'], _n, p=[0.55,0.2,0.2,0.05]).tolist()
for _i in np.random.choice(_n, 10, replace=False): _emp[_i] = None

_ct = np.random.choice(['Tier1','Tier2','Tier3'], _n).tolist()
for _i in np.random.choice(_n, 8, replace=False): _ct[_i] = None

_df = pd.DataFrame({
    'age':             _age,
    'income':          _income,
    'credit_score':    _cs,
    'loan_amount':     _la,
    'employment_type': _emp,
    'city_tier':       _ct,
    'loan_tenure':     np.random.choice([12,24,36,48,60], _n),
    'is_defaulter':    np.random.choice([0,1], _n, p=[0.78,0.22]),
})
_df['loan_amount_str'] = _df['loan_amount'].apply(lambda x: str(int(x)) if not pd.isna(x) else None)

_G = {"df": _df.copy(), "pd": pd, "np": np}

DF_SETUP = """\
import pandas as pd, numpy as np
np.random.seed(202)
_n = 300
_age = np.random.randint(22, 65, _n).astype(float)
_age[np.random.choice(_n, 15, replace=False)] = np.nan
_income = np.random.randint(200000, 2500000, _n).astype(float)
_income[np.random.choice(_n, 12, replace=False)] = np.nan
_income[np.random.choice(_n, 5, replace=False)] = np.random.randint(8000000, 12000000, 5)
_cs = np.random.randint(500, 850, _n).astype(float)
_cs[np.random.choice(_n, 10, replace=False)] = np.nan
_la = np.random.randint(100000, 2000000, _n).astype(float)
_la[np.random.choice(_n, 8, replace=False)] = np.nan
_emp = np.random.choice(['Salaried','Self-Employed','Business','Unemployed'], _n, p=[0.55,0.2,0.2,0.05]).tolist()
for _i in np.random.choice(_n, 10, replace=False): _emp[_i] = None
_ct = np.random.choice(['Tier1','Tier2','Tier3'], _n).tolist()
for _i in np.random.choice(_n, 8, replace=False): _ct[_i] = None
df = pd.DataFrame({
    'age':             _age,
    'income':          _income,
    'credit_score':    _cs,
    'loan_amount':     _la,
    'employment_type': _emp,
    'city_tier':       _ct,
    'loan_tenure':     np.random.choice([12,24,36,48,60], _n),
    'is_defaulter':    np.random.choice([0,1], _n, p=[0.78,0.22]),
})
df['loan_amount_str'] = df['loan_amount'].apply(lambda x: str(int(x)) if not pd.isna(x) else None)
"""

# ── PRECOMPUTE ALL ANSWERS ────────────────────────────────────────

# MCQ / fill
_total_missing   = int(_df.isnull().sum().sum())
_age_missing     = int(_df['age'].isnull().sum())
_income_median   = round(_df['income'].median(), 2)
_iqr_cs          = round(_df['credit_score'].quantile(0.75) - _df['credit_score'].quantile(0.25), 2)
_income_3std     = len(_df[abs(_df['income'] - _df['income'].mean()) > 3*_df['income'].std()])
_emp_mode        = _df['employment_type'].mode()[0]
_ct_mode         = _df['city_tier'].mode()[0]
_emp_nunique     = _df['employment_type'].nunique()

# Code answers
_p1e7 = _exec("df2=df.copy()\ndf2['age']=df2['age'].fillna(df2['age'].median())\nprint(df2['age'].isnull().sum())", {**_G,"df":_df.copy()})
_p1e8 = _exec("print(df.isnull().sum().idxmax())", _G)
_p1e9 = _exec("print(df.dropna().shape[0])", _G)

_p2e7 = _exec("q1=df['income'].quantile(0.25)\nq3=df['income'].quantile(0.75)\niqr=q3-q1\nprint(len(df[(df['income']<q1-1.5*iqr)|(df['income']>q3+1.5*iqr)]))", _G)
_p2e8 = _exec("df2=df.copy()\ndf2['income']=df2['income'].clip(lower=df2['income'].quantile(0.05),upper=df2['income'].quantile(0.95))\nprint(round(df2['income'].max(),2))", {**_G,"df":_df.copy()})
_p2e9 = _exec("df2=df.copy()\ndf2['income_zscore']=((df2['income']-df2['income'].mean())/df2['income'].std()).round(2)\nprint(df2['income_zscore'].abs().max())", {**_G,"df":_df.copy()})

_p3e7 = _exec("print(df['employment_type'].value_counts().idxmax())", _G)
_p3e8 = _exec("dummies=pd.get_dummies(df['city_tier'],prefix='city',drop_first=True,dtype=int)\nprint(list(dummies.columns))", _G)
_p3e9 = _exec("from sklearn.preprocessing import MinMaxScaler\ndf2=df.dropna(subset=['income'])\nscaler=MinMaxScaler()\nscaled=scaler.fit_transform(df2[['income']])\nprint(round(float(scaled.max()),2))", _G)

_fe1  = str(_total_missing)
_fe2  = _df.isnull().sum().idxmax()
_fe3  = _exec("df2=df.copy()\ndf2['age']=df2['age'].fillna(df2['age'].median())\ndf2['income']=df2['income'].fillna(df2['income'].median())\ndf2['credit_score']=df2['credit_score'].fillna(df2['credit_score'].median())\ndf2['loan_amount']=df2['loan_amount'].fillna(df2['loan_amount'].median())\ndf2['employment_type']=df2['employment_type'].fillna(df2['employment_type'].mode()[0])\ndf2['city_tier']=df2['city_tier'].fillna(df2['city_tier'].mode()[0])\nprint(df2.isnull().sum().sum())", {**_G,"df":_df.copy()})
_fe4  = _exec("q1=df['income'].quantile(0.25)\nq3=df['income'].quantile(0.75)\niqr=q3-q1\nprint(len(df[(df['income']<q1-1.5*iqr)|(df['income']>q3+1.5*iqr)]))", _G)
_fe5  = _exec("df2=df.copy()\ndf2['loan_amount']=pd.to_numeric(df2['loan_amount_str'],errors='coerce')\nprint(df2['loan_amount'].dtype)", {**_G,"df":_df.copy()})
_fe6  = _exec("df2=df.copy()\ndf2['income_per_loan']=df2['income']/df2['loan_amount']\nprint(round(df2['income_per_loan'].mean(),2))", {**_G,"df":_df.copy()})
_fe7  = _exec("dummies=pd.get_dummies(df['employment_type'],prefix='emp',drop_first=True,dtype=int)\nprint(list(dummies.columns))", _G)
_fe8  = _exec("from sklearn.preprocessing import StandardScaler\ndf2=df.dropna(subset=['income','loan_amount'])\nscaler=StandardScaler()\nscaled=scaler.fit_transform(df2[['income','loan_amount']])\nprint(round(float(scaled[:,0].mean()),2))", _G)
_fe9  = _exec("df2=df.copy()\ndf2['risk_segment']=pd.cut(df2['credit_score'],bins=[0,580,700,850],labels=['High Risk','Medium Risk','Low Risk'])\nprint(df2['risk_segment'].value_counts().idxmax())", {**_G,"df":_df.copy()})
_fe10 = _exec("df2=df.copy()\ndf2['age']=df2['age'].fillna(df2['age'].median())\ndf2['age_group']=pd.cut(df2['age'],bins=[0,30,45,100],labels=['Young','Mid','Senior'])\nprint(df2['age_group'].value_counts().idxmax())", {**_G,"df":_df.copy()})
_fe11 = _exec("from sklearn.preprocessing import MinMaxScaler\ndf2=df.dropna(subset=['income']).copy()\nscaler=MinMaxScaler()\ndf2['income_scaled']=scaler.fit_transform(df2[['income']])\nprint(round(df2['income_scaled'].mean(),2))", {**_G,"df":_df.copy()})
_fe12 = _exec("df2=df.copy()\ndf2['high_risk']=((df2['credit_score']<580)&(df2['is_defaulter']==1)).astype(int)\nprint(df2['high_risk'].sum())", {**_G,"df":_df.copy()})

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 1 — Missing Value Handling
# ════════════════════════════════════════════════════════════════
M2_PRACTICE_1 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"Which strategy is best for imputing missing values in a heavily skewed numeric column?",
     "opts":["Mean imputation","Median imputation","Zero imputation","Drop all rows"],
     "ah":h("Median imputation"),
     "solution":"# Mean is sensitive to outliers in skewed distributions\n# Median is robust to skew\ndf['income'] = df['income'].fillna(df['income'].median())",
     "explanation":"Median is unaffected by extreme values (outliers) unlike mean, making it better for skewed distributions like income."},
    {"id":2,"type":"mcq","marks":2,
     "text":"What does df.isnull().sum() return?",
     "opts":[
         "Total missing values in entire DataFrame",
         "Count of missing values per column",
         "Boolean DataFrame of missing flags",
         "Rows with any missing value"
     ],
     "ah":h("Count of missing values per column"),
     "solution":"import pandas as pd\n# df.isnull() → Boolean DataFrame\n# .sum() → sums True (missing) per column\nprint(df.isnull().sum())",
     "explanation":"isnull() returns True/False per cell. sum() aggregates per column giving count of missing values per column."},
    {"id":3,"type":"fill","marks":2,
     "text":"How many total missing values are in the dataset?\n\ndf.isnull().sum().sum()\n\nType the integer.",
     "ah":h(str(_total_missing)),
     "exp":str(_total_missing),
     "solution":f"print(df.isnull().sum().sum())  # {_total_missing}",
     "explanation":f"Total missing across all columns = {_total_missing}. Use .sum().sum() — first sum per column, then sum all columns."},
    {"id":4,"type":"fill","marks":2,
     "text":"How many missing values are in the 'age' column?\n\nType the integer.",
     "ah":h(str(_age_missing)),
     "exp":str(_age_missing),
     "solution":f"print(df['age'].isnull().sum())  # {_age_missing}",
     "explanation":f"The age column has {_age_missing} missing values introduced during dataset creation."},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the median income of all customers (including NaN rows)?\n\ndf['income'].median()\n\nType the exact value.",
     "ah":h(str(_income_median)),
     "exp":str(_income_median),
     "solution":f"print(df['income'].median())  # {_income_median}",
     "explanation":"pandas median() automatically skips NaN values. Result is the middle value of non-null income values."},
    {"id":6,"type":"mcq","marks":3,
     "text":"When should you use forward fill (ffill) for missing values?",
     "opts":[
         "When data is randomly missing in any column",
         "When data is time-series or sequential and previous value is a valid estimate",
         "When the column has more than 50% missing values",
         "When imputing categorical columns"
     ],
     "ah":h("When data is time-series or sequential and previous value is a valid estimate"),
     "solution":"# ffill propagates last valid observation forward\ndf['price'] = df['price'].fillna(method='ffill')\n# Best for: stock prices, sensor readings, sequential data",
     "explanation":"ffill (forward fill) uses the previous non-null value. Only valid when data is ordered and temporal continuity makes sense."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nFill missing values in the 'age' column with the median age.\nPrint the number of missing values in 'age' after filling.",
     "preload":DF_SETUP,
     "exp":_p1e7,
     "solution":"df2 = df.copy()\ndf2['age'] = df2['age'].fillna(df2['age'].median())\nprint(df2['age'].isnull().sum())  # 0",
     "explanation":"fillna with median fills all NaN positions. After filling, null count should be 0."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nFind which column has the highest number of missing values.\nPrint only the column name.",
     "preload":DF_SETUP,
     "exp":_p1e8,
     "solution":f"print(df.isnull().sum().idxmax())  # {_p1e8}",
     "explanation":"isnull().sum() gives missing count per column. idxmax() returns the column name with highest count."},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (300 rows).\n\nDrop all rows that have ANY missing value.\nPrint the number of remaining rows.",
     "preload":DF_SETUP,
     "exp":_p1e9,
     "solution":f"print(df.dropna().shape[0])  # {_p1e9}",
     "explanation":"dropna() by default drops any row with at least one NaN. shape[0] gives the row count."},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 2 — Outlier Detection
# ════════════════════════════════════════════════════════════════
M2_PRACTICE_2 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"In the IQR method, a value is flagged as an outlier if it falls:",
     "opts":[
         "Beyond mean ± 2 standard deviations",
         "Below Q1 - 1.5*IQR or above Q3 + 1.5*IQR",
         "Below the 5th percentile or above the 95th percentile",
         "More than 3 standard deviations from median"
     ],
     "ah":h("Below Q1 - 1.5*IQR or above Q3 + 1.5*IQR"),
     "solution":"Q1 = df['col'].quantile(0.25)\nQ3 = df['col'].quantile(0.75)\nIQR = Q3 - Q1\noutliers = df[(df['col'] < Q1-1.5*IQR) | (df['col'] > Q3+1.5*IQR)]",
     "explanation":"The Tukey fence method uses 1.5*IQR below Q1 and above Q3 to define outlier boundaries."},
    {"id":2,"type":"mcq","marks":2,
     "text":"What is the main advantage of using IQR over Z-score for outlier detection?",
     "opts":[
         "IQR is faster to compute",
         "IQR is robust to extreme values and does not assume normal distribution",
         "IQR always detects more outliers",
         "IQR works only on categorical data"
     ],
     "ah":h("IQR is robust to extreme values and does not assume normal distribution"),
     "solution":"# Z-score assumes normality and is affected by extreme values\n# IQR uses quartiles which are resistant to outliers\n# Use IQR for skewed data, Z-score for normally distributed data",
     "explanation":"Z-score uses mean and std which are sensitive to outliers. IQR uses Q1/Q3 which are not affected by extremes."},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the IQR of credit_score in the dataset?\n\nIQR = Q3 - Q1\nType rounded to 2 decimal places.",
     "ah":h(str(_iqr_cs)),
     "exp":str(_iqr_cs),
     "solution":f"Q1 = df['credit_score'].quantile(0.25)\nQ3 = df['credit_score'].quantile(0.75)\nprint(round(Q3-Q1, 2))  # {_iqr_cs}",
     "explanation":f"IQR of credit_score = Q3 - Q1 = {_iqr_cs}. Measures spread of middle 50% of credit scores."},
    {"id":4,"type":"fill","marks":3,
     "text":"How many income outliers exist using the 3-standard-deviation rule?\n\n(Values where |income - mean| > 3 * std)\n\nType the integer.",
     "ah":h(str(_income_3std)),
     "exp":str(_income_3std),
     "solution":f"outliers = df[abs(df['income']-df['income'].mean()) > 3*df['income'].std()]\nprint(len(outliers))  # {_income_3std}",
     "explanation":f"Z-score > 3 flags extreme outliers. The dataset has {_income_3std} such values in income — the artificially injected high values."},
    {"id":5,"type":"mcq","marks":3,
     "text":"What is 'winsorization' in the context of outlier treatment?",
     "opts":[
         "Removing all outlier rows from the dataset",
         "Replacing outliers with the nearest boundary value (e.g. 5th/95th percentile)",
         "Transforming outliers using log transformation",
         "Flagging outliers with a binary indicator column"
     ],
     "ah":h("Replacing outliers with the nearest boundary value (e.g. 5th/95th percentile)"),
     "solution":"# Winsorization: clip values to percentile boundaries\ndf['income'] = df['income'].clip(\n    lower=df['income'].quantile(0.05),\n    upper=df['income'].quantile(0.95)\n)",
     "explanation":"Winsorization replaces extreme values with boundary values rather than dropping rows, preserving dataset size."},
    {"id":6,"type":"mcq","marks":3,
     "text":"Which visualization is BEST for visually spotting outliers in a numeric column?",
     "opts":["Histogram","Box plot","Bar chart","Pie chart"],
     "ah":h("Box plot"),
     "solution":"import matplotlib.pyplot as plt\ndf['income'].plot(kind='box')\nplt.show()\n# Box plot shows Q1, Q3, whiskers and outlier points clearly",
     "explanation":"Box plots display IQR, whiskers (1.5*IQR), and dots for individual outliers — purpose-built for outlier visualization."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nUsing the IQR method, count how many income values are outliers.\n(Below Q1-1.5*IQR or above Q3+1.5*IQR)\nPrint the count.",
     "preload":DF_SETUP,
     "exp":_p2e7,
     "solution":f"q1 = df['income'].quantile(0.25)\nq3 = df['income'].quantile(0.75)\niqr = q3 - q1\noutliers = df[(df['income']<q1-1.5*iqr)|(df['income']>q3+1.5*iqr)]\nprint(len(outliers))  # {_p2e7}",
     "explanation":"IQR fence: lower = Q1-1.5*IQR, upper = Q3+1.5*IQR. Count rows outside these bounds."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nWinsorize the income column by clipping values to\nthe 5th and 95th percentiles.\nPrint the maximum income value after clipping, rounded to 2 decimal places.",
     "preload":DF_SETUP,
     "exp":_p2e8,
     "solution":f"df2 = df.copy()\ndf2['income'] = df2['income'].clip(\n    lower=df2['income'].quantile(0.05),\n    upper=df2['income'].quantile(0.95)\n)\nprint(round(df2['income'].max(), 2))  # {_p2e8}",
     "explanation":"clip() replaces values outside bounds with the boundary value. max() after clipping = 95th percentile value."},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (300 rows).\n\nCompute the Z-score of the income column.\nRound each Z-score to 2 decimal places.\nPrint the maximum absolute Z-score.",
     "preload":DF_SETUP,
     "exp":_p2e9,
     "solution":f"df2 = df.copy()\ndf2['income_zscore'] = ((df2['income']-df2['income'].mean())/df2['income'].std()).round(2)\nprint(df2['income_zscore'].abs().max())  # {_p2e9}",
     "explanation":"Z-score = (value - mean) / std. A Z-score > 3 indicates an extreme outlier. The max absolute Z-score reveals the most extreme income."},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 3 — Encoding + Scaling + New Variables
# ════════════════════════════════════════════════════════════════
M2_PRACTICE_3 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"When should you use one-hot encoding instead of label encoding?",
     "opts":[
         "When the categorical variable has a natural order (e.g. Low/Medium/High)",
         "When the categorical variable has no ordinal relationship (nominal categories)",
         "When the column has more than 10 unique values",
         "When the model is a decision tree"
     ],
     "ah":h("When the categorical variable has no ordinal relationship (nominal categories)"),
     "solution":"# Label encoding: Low=0, Medium=1, High=2 (ordinal - order matters)\n# One-hot encoding: City=[Mumbai, Delhi, Bangalore] → 3 binary columns\n# Use one-hot when categories have no ranking",
     "explanation":"Label encoding implies an order between categories. For nominal data (city, employment type), one-hot encoding avoids false ordinal relationships."},
    {"id":2,"type":"mcq","marks":2,
     "text":"What is the difference between Min-Max scaling and Standard scaling (Z-score)?",
     "opts":[
         "Min-Max scales to [0,1]; Standard scaling gives mean=0, std=1",
         "Min-Max removes outliers; Standard scaling does not",
         "Standard scaling scales to [0,1]; Min-Max gives mean=0",
         "They produce identical results"
     ],
     "ah":h("Min-Max scales to [0,1]; Standard scaling gives mean=0, std=1"),
     "solution":"from sklearn.preprocessing import MinMaxScaler, StandardScaler\n# MinMaxScaler: x_scaled = (x - min) / (max - min) → range [0,1]\n# StandardScaler: x_scaled = (x - mean) / std → mean=0, std=1",
     "explanation":"MinMaxScaler compresses to [0,1]. StandardScaler centers at 0 with unit variance. Use StandardScaler when data is approximately normal."},
    {"id":3,"type":"fill","marks":3,
     "text":"Which employment_type is most common in the dataset?\n\ndf['employment_type'].value_counts().idxmax()\n\nType the exact string.",
     "ah":h(_p3e7),
     "exp":_p3e7,
     "solution":f"print(df['employment_type'].value_counts().idxmax())  # {_p3e7}",
     "explanation":f"value_counts() sorts by frequency. idxmax() returns the category name with highest count = {_p3e7}."},
    {"id":4,"type":"fill","marks":3,
     "text":"How many unique employment types are there (excluding NaN)?\n\ndf['employment_type'].nunique()\n\nType the integer.",
     "ah":h(str(_emp_nunique)),
     "exp":str(_emp_nunique),
     "solution":f"print(df['employment_type'].nunique())  # {_emp_nunique}",
     "explanation":f"nunique() counts distinct non-null values. There are {_emp_nunique} employment types: Salaried, Self-Employed, Business, Unemployed."},
    {"id":5,"type":"mcq","marks":3,
     "text":"What does drop_first=True do in pd.get_dummies()?",
     "opts":[
         "Removes the first row of the DataFrame",
         "Drops the first dummy column to avoid multicollinearity",
         "Removes columns with zero variance",
         "Drops NaN rows before encoding"
     ],
     "ah":h("Drops the first dummy column to avoid multicollinearity"),
     "solution":"# Without drop_first: 3 categories → 3 dummy columns (redundant)\n# With drop_first: 3 categories → 2 dummy columns\n# The dropped column is implied when all others are 0\ndummies = pd.get_dummies(df['city_tier'], drop_first=True)",
     "explanation":"With k categories, k-1 dummy columns are sufficient. The reference category is implied when all dummies = 0. Avoids the dummy variable trap."},
    {"id":6,"type":"mcq","marks":3,
     "text":"When creating a new variable 'income_to_loan_ratio = income / loan_amount', what should you do first?",
     "opts":[
         "Scale both columns first",
         "Handle missing values in both columns first",
         "Encode the columns",
         "Drop outliers first"
     ],
     "ah":h("Handle missing values in both columns first"),
     "solution":"# Step 1: Handle NaN — division with NaN produces NaN\ndf2 = df.copy()\ndf2['income'] = df2['income'].fillna(df2['income'].median())\ndf2['loan_amount'] = df2['loan_amount'].fillna(df2['loan_amount'].median())\n# Step 2: Create ratio\ndf2['income_to_loan_ratio'] = df2['income'] / df2['loan_amount']",
     "explanation":"Any NaN in income or loan_amount will propagate to the new ratio column. Always handle missing values before feature creation."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nFind the most common employment_type (excluding NaN).\nPrint only the employment type name.",
     "preload":DF_SETUP,
     "exp":_p3e7,
     "solution":f"print(df['employment_type'].value_counts().idxmax())  # {_p3e7}",
     "explanation":"value_counts() automatically ignores NaN. idxmax() returns the label with highest frequency."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (300 rows).\n\nApply one-hot encoding to the 'city_tier' column using pd.get_dummies().\nUse prefix='city', drop_first=True, dtype=int.\nPrint the list of new dummy column names.",
     "preload":DF_SETUP,
     "exp":_p3e8,
     "solution":f"dummies = pd.get_dummies(df['city_tier'], prefix='city', drop_first=True, dtype=int)\nprint(list(dummies.columns))  # {_p3e8}",
     "explanation":"With 3 city tiers and drop_first=True, we get 2 dummy columns. The first tier (Tier1) becomes the reference category."},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (300 rows).\n\nApply Min-Max scaling to the 'income' column.\nDrop NaN rows in income first before scaling.\nPrint the maximum scaled value rounded to 2 decimal places.",
     "preload":DF_SETUP,
     "exp":_p3e9,
     "solution":f"from sklearn.preprocessing import MinMaxScaler\ndf2 = df.dropna(subset=['income'])\nscaler = MinMaxScaler()\nscaled = scaler.fit_transform(df2[['income']])\nprint(round(float(scaled.max()), 2))  # {_p3e9}",
     "explanation":"MinMaxScaler transforms to [0, 1]. The maximum scaled value is always 1.0 (the original maximum maps to 1)."},
]

# ════════════════════════════════════════════════════════════════
# FINAL ASSESSMENT — Moderate, 45 min, 1 attempt
# ════════════════════════════════════════════════════════════════
M2_FINAL = [
    {"id":1,"type":"code","marks":3,
     "text":"df is preloaded (300 rows, 9 columns).\nColumns: age, income, credit_score, loan_amount,\n         employment_type, city_tier, loan_tenure,\n         is_defaulter, loan_amount_str\n\nHow many total missing values are in the entire dataset?\nPrint the integer.",
     "preload":DF_SETUP,
     "exp":_fe1,
     "solution":f"print(df.isnull().sum().sum())  # {_fe1}",
     "explanation":f"Total missing = {_fe1}. Use .isnull().sum().sum() — first sum per column, then sum all columns together."},
    {"id":2,"type":"code","marks":3,
     "text":"df is preloaded.\n\nWhich column has the most missing values?\nPrint only the column name.",
     "preload":DF_SETUP,
     "exp":_fe2,
     "solution":f"print(df.isnull().sum().idxmax())  # {_fe2}",
     "explanation":f"isnull().sum() counts NaN per column. idxmax() returns the column with the maximum count = '{_fe2}'."},
    {"id":3,"type":"code","marks":4,
     "text":"df is preloaded.\n\nFill missing values as follows:\n- Numeric columns (age, income, credit_score, loan_amount): fill with median\n- Categorical columns (employment_type, city_tier): fill with mode\n\nPrint total missing values after filling.\n(Note: loan_amount_str will still have NaN — ignore it)",
     "preload":DF_SETUP,
     "exp":_fe3,
     "solution":f"df2 = df.copy()\ndf2['age'] = df2['age'].fillna(df2['age'].median())\ndf2['income'] = df2['income'].fillna(df2['income'].median())\ndf2['credit_score'] = df2['credit_score'].fillna(df2['credit_score'].median())\ndf2['loan_amount'] = df2['loan_amount'].fillna(df2['loan_amount'].median())\ndf2['employment_type'] = df2['employment_type'].fillna(df2['employment_type'].mode()[0])\ndf2['city_tier'] = df2['city_tier'].fillna(df2['city_tier'].mode()[0])\nprint(df2.isnull().sum().sum())  # {_fe3}",
     "explanation":f"After filling all numeric and categorical columns, only loan_amount_str (derived from loan_amount) still has NaN → {_fe3} remaining."},
    {"id":4,"type":"code","marks":4,
     "text":"df is preloaded.\n\nUsing the IQR method, count outliers in the income column.\n(Below Q1-1.5*IQR or above Q3+1.5*IQR)\nPrint the count.",
     "preload":DF_SETUP,
     "exp":_fe4,
     "solution":f"q1 = df['income'].quantile(0.25)\nq3 = df['income'].quantile(0.75)\niqr = q3 - q1\noutliers = df[(df['income'] < q1-1.5*iqr) | (df['income'] > q3+1.5*iqr)]\nprint(len(outliers))  # {_fe4}",
     "explanation":f"IQR fence detects {_fe4} income outliers — these are the artificially injected high-value incomes."},
    {"id":5,"type":"code","marks":4,
     "text":"df is preloaded.\n\nConvert the 'loan_amount_str' column to numeric using pd.to_numeric().\nUse errors='coerce' to handle non-numeric values.\nPrint the dtype of the resulting column.",
     "preload":DF_SETUP,
     "exp":_fe5,
     "solution":f"df2 = df.copy()\ndf2['loan_amount'] = pd.to_numeric(df2['loan_amount_str'], errors='coerce')\nprint(df2['loan_amount'].dtype)  # {_fe5}",
     "explanation":f"pd.to_numeric() with errors='coerce' converts valid strings to numbers and invalid ones to NaN. Result dtype = {_fe5}."},
    {"id":6,"type":"code","marks":4,
     "text":"df is preloaded.\n\nCreate a new column 'income_per_loan' = income / loan_amount.\nPrint the mean of this new column rounded to 2 decimal places.",
     "preload":DF_SETUP,
     "exp":_fe6,
     "solution":f"df2 = df.copy()\ndf2['income_per_loan'] = df2['income'] / df2['loan_amount']\nprint(round(df2['income_per_loan'].mean(), 2))  # {_fe6}",
     "explanation":f"Dividing income by loan_amount creates a ratio feature. NaN in either column propagates to result. Mean = {_fe6}."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded.\n\nApply one-hot encoding to 'employment_type' using pd.get_dummies().\nUse prefix='emp', drop_first=True, dtype=int.\nPrint the list of new dummy column names.",
     "preload":DF_SETUP,
     "exp":_fe7,
     "solution":f"dummies = pd.get_dummies(df['employment_type'], prefix='emp', drop_first=True, dtype=int)\nprint(list(dummies.columns))  # {_fe7}",
     "explanation":f"4 employment types with drop_first=True → 3 dummy columns. 'Business' becomes the reference (dropped) category."},
    {"id":8,"type":"code","marks":5,
     "text":"df is preloaded.\n\nApply Standard Scaling (StandardScaler) to both 'income' and 'loan_amount'.\nDrop rows with NaN in either column first.\nPrint the mean of the scaled income column, rounded to 2 decimal places.",
     "preload":DF_SETUP,
     "exp":_fe8,
     "solution":f"from sklearn.preprocessing import StandardScaler\ndf2 = df.dropna(subset=['income', 'loan_amount'])\nscaler = StandardScaler()\nscaled = scaler.fit_transform(df2[['income', 'loan_amount']])\nprint(round(float(scaled[:, 0].mean()), 2))  # {_fe8}",
     "explanation":f"StandardScaler centers data at mean=0. After scaling, the mean of income column = {_fe8} (effectively 0)."},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded.\n\nCreate a 'risk_segment' column by binning credit_score:\n  bins: [0, 580, 700, 850]\n  labels: ['High Risk', 'Medium Risk', 'Low Risk']\n\nPrint the most common risk segment.",
     "preload":DF_SETUP,
     "exp":_fe9,
     "solution":f"df2 = df.copy()\ndf2['risk_segment'] = pd.cut(\n    df2['credit_score'],\n    bins=[0, 580, 700, 850],\n    labels=['High Risk', 'Medium Risk', 'Low Risk']\n)\nprint(df2['risk_segment'].value_counts().idxmax())  # {_fe9}",
     "explanation":f"pd.cut() bins continuous values into categories. Most customers fall in '{_fe9}' segment based on credit score distribution."},
    {"id":10,"type":"code","marks":5,
     "text":"df is preloaded.\n\nFirst fill missing 'age' values with median.\nThen create 'age_group' using pd.cut():\n  bins: [0, 30, 45, 100]\n  labels: ['Young', 'Mid', 'Senior']\n\nPrint the most common age group.",
     "preload":DF_SETUP,
     "exp":_fe10,
     "solution":f"df2 = df.copy()\ndf2['age'] = df2['age'].fillna(df2['age'].median())\ndf2['age_group'] = pd.cut(\n    df2['age'],\n    bins=[0, 30, 45, 100],\n    labels=['Young', 'Mid', 'Senior']\n)\nprint(df2['age_group'].value_counts().idxmax())  # {_fe10}",
     "explanation":f"Fill NaN first so no age values are lost in binning. Most customers are in the '{_fe10}' age group (30-45 years)."},
    {"id":11,"type":"code","marks":5,
     "text":"df is preloaded.\n\nApply Min-Max scaling to the 'income' column.\nDrop NaN rows in income first.\nStore scaled values in a new column 'income_scaled'.\nPrint the mean of 'income_scaled' rounded to 2 decimal places.",
     "preload":DF_SETUP,
     "exp":_fe11,
     "solution":f"from sklearn.preprocessing import MinMaxScaler\ndf2 = df.dropna(subset=['income']).copy()\nscaler = MinMaxScaler()\ndf2['income_scaled'] = scaler.fit_transform(df2[['income']])\nprint(round(df2['income_scaled'].mean(), 2))  # {_fe11}",
     "explanation":f"MinMaxScaler transforms income to [0,1]. Mean of scaled income = {_fe11} — not 0.5 because distribution is not symmetric."},
    {"id":12,"type":"mcq","marks":3,
     "text":"Which of the following is the correct order for a data preprocessing pipeline?",
     "opts":[
         "Scale → Encode → Handle Missing → Create Features",
         "Handle Missing → Encode → Create Features → Scale",
         "Encode → Scale → Handle Missing → Create Features",
         "Create Features → Handle Missing → Scale → Encode"
     ],
     "ah":h("Handle Missing → Encode → Create Features → Scale"),
     "solution":"# Correct order:\n# 1. Handle missing values (impute/drop)\n# 2. Encode categoricals (get_dummies / label encode)\n# 3. Create new features (ratios, bins, interactions)\n# 4. Scale numeric features (MinMax / Standard)",
     "explanation":"Missing values must be handled first. Encoding converts categoricals. New features can use encoded/imputed values. Scaling is done last — it uses numeric values only."},
]

# ── MAPS ─────────────────────────────────────────────────────────
M2_PRACTICE_TESTS = {1: M2_PRACTICE_1, 2: M2_PRACTICE_2, 3: M2_PRACTICE_3}
M2_FINAL_TEST     = {1: M2_FINAL}
M2_DATASET        = _df.copy()
