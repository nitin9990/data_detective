"""
questions_m3.py — Module 3: Descriptive Statistics + EDA + Visualization
3 Practice sets + 1 Final assessment
Banking/Financial dataset (seed=101, 400 rows)
"""

import hashlib, re, io, sys
import numpy as np
import pandas as pd

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

def _exec(code, globs):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    exec(code, globs); sys.stdout = old
    return buf.getvalue().strip()

# ── DATASET (fixed seed=101) ──────────────────────────────────────
np.random.seed(101); _n = 400
_df = pd.DataFrame({
    'customer_id':    range(6001, 6401),
    'age':            np.random.randint(22, 65, _n),
    'income':         np.random.randint(200000, 3000000, _n),
    'credit_score':   np.random.randint(450, 900, _n),
    'loan_amount':    np.random.randint(50000, 2000000, _n),
    'loan_tenure':    np.random.choice([12, 24, 36, 48, 60], _n),
    'emi_amount':     np.random.randint(5000, 80000, _n),
    'num_products':   np.random.randint(1, 6, _n),
    'is_defaulter':   np.random.choice([0, 1], _n, p=[0.78, 0.22]),
    'gender':         np.random.choice(['M', 'F'], _n, p=[0.6, 0.4]),
    'region':         np.random.choice(['North', 'South', 'East', 'West'], _n),
    'months_with_bank': np.random.randint(6, 120, _n),
})
_G = {"df": _df.copy(), "pd": pd, "np": np}

DF_SETUP = """\
import pandas as pd, numpy as np
np.random.seed(101)
_n = 400
df = pd.DataFrame({
    'customer_id':    range(6001, 6401),
    'age':            np.random.randint(22, 65, _n),
    'income':         np.random.randint(200000, 3000000, _n),
    'credit_score':   np.random.randint(450, 900, _n),
    'loan_amount':    np.random.randint(50000, 2000000, _n),
    'loan_tenure':    np.random.choice([12, 24, 36, 48, 60], _n),
    'emi_amount':     np.random.randint(5000, 80000, _n),
    'num_products':   np.random.randint(1, 6, _n),
    'is_defaulter':   np.random.choice([0, 1], _n, p=[0.78, 0.22]),
    'gender':         np.random.choice(['M', 'F'], _n, p=[0.6, 0.4]),
    'region':         np.random.choice(['North', 'South', 'East', 'West'], _n),
    'months_with_bank': np.random.randint(6, 120, _n),
})
"""

# Precompute all MCQ/fill answers
_defaulter_rate  = round(_df['is_defaulter'].mean(), 2)
_median_cs       = _df['credit_score'].median()
_mean_income     = round(_df['income'].mean(), 2)
_skew_income     = round(_df['income'].skew(), 2)
_region_max      = _df['region'].value_counts().idxmax()
_corr_cs_def     = round(_df['credit_score'].corr(_df['is_defaulter']), 2)
_iqr_loan        = round(_df['loan_amount'].quantile(0.75) - _df['loan_amount'].quantile(0.25), 2)
_highest_def_reg = _df.groupby('region')['is_defaulter'].mean().idxmax()
_q1_cs           = _df['credit_score'].quantile(0.25)
_q3_cs           = _df['credit_score'].quantile(0.75)
_gender_loan_max = _df.groupby('gender')['loan_amount'].mean().idxmax()

# Precompute code expected outputs
_p1e7  = _exec("print(df['is_defaulter'].value_counts().to_dict())", {**_G, "df": _df.copy()})
_p1e8  = _exec("print(round(df['income'].skew(), 2))", {**_G, "df": _df.copy()})
_p1e9  = _exec("print(df.isnull().sum().sum())", {**_G, "df": _df.copy()})

_p2e7  = _exec("print(df.groupby('region')['is_defaulter'].mean().idxmax())", {**_G, "df": _df.copy()})
_p2e8  = _exec("print(round(df['loan_amount'].quantile(0.75)-df['loan_amount'].quantile(0.25), 2))", {**_G, "df": _df.copy()})
_p2e9  = _exec("outliers=df[abs(df['credit_score']-df['credit_score'].mean())>2*df['credit_score'].std()]\nprint(len(outliers))", {**_G, "df": _df.copy()})

_p3e7  = _exec("print(round(df['credit_score'].corr(df['is_defaulter']), 2))", {**_G, "df": _df.copy()})
_p3e8  = _exec("print(df.groupby('gender')['loan_amount'].mean().idxmax())", {**_G, "df": _df.copy()})
_p3e9  = _exec("bins=pd.cut(df['credit_score'],bins=[0,550,700,900],labels=['Low','Medium','High'])\nprint(bins.value_counts().idxmax())", {**_G, "df": _df.copy()})

_fe1   = _exec("print(round(df['income'].mean(), 2))", {**_G, "df": _df.copy()})
_fe2   = _exec("print(df['credit_score'].median())", {**_G, "df": _df.copy()})
_fe3   = _exec("print(df.groupby('region')['is_defaulter'].mean().round(2).to_dict())", {**_G, "df": _df.copy()})
_fe4   = _exec("q1=df['loan_amount'].quantile(0.25)\nq3=df['loan_amount'].quantile(0.75)\niqr=q3-q1\nprint(len(df[(df['loan_amount']<q1-1.5*iqr)|(df['loan_amount']>q3+1.5*iqr)]))", {**_G, "df": _df.copy()})
_fe5   = _exec("cols=['income','credit_score','loan_amount','emi_amount']\ncorr_vals={c:abs(df[c].corr(df['is_defaulter'])) for c in cols}\nprint(max(corr_vals,key=corr_vals.get))", {**_G, "df": _df.copy()})
_fe6   = _exec("print(df.groupby('loan_tenure')['is_defaulter'].mean().idxmax())", {**_G, "df": _df.copy()})
_fe7   = _exec("print(round(df[df['is_defaulter']==1]['credit_score'].mean()-df[df['is_defaulter']==0]['credit_score'].mean(),2))", {**_G, "df": _df.copy()})
_fe8   = _exec("df2=df.copy()\ndf2['income_band']=pd.cut(df2['income'],bins=3,labels=['Low','Medium','High'])\nprint(df2.groupby('income_band',observed=True)['is_defaulter'].mean().idxmax())", {**_G, "df": _df.copy()})
_fe9   = _exec("print(df[(df['credit_score']<550)&(df['is_defaulter']==0)].shape[0])", {**_G, "df": _df.copy()})
_fe10  = _exec("print(round(df['emi_amount'].std(), 2))", {**_G, "df": _df.copy()})
_fe11  = _exec("print(df.groupby('num_products')['is_defaulter'].mean().round(2).to_dict())", {**_G, "df": _df.copy()})
_fe12  = _exec("print(round(df[df['is_defaulter']==1]['loan_amount'].mean(), 2))", {**_G, "df": _df.copy()})

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 1 — Descriptive Statistics
# ════════════════════════════════════════════════════════════════
M3_PRACTICE_1 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"A distribution has mean=500, median=350. What does this suggest?",
     "opts":[
         "Symmetric distribution",
         "Negatively skewed — tail on the left",
         "Positively skewed — tail on the right",
         "Bimodal distribution"
     ],
     "ah":h("Positively skewed — tail on the right")},
    {"id":2,"type":"mcq","marks":2,
     "text":"Which measure of central tendency is most resistant to outliers?",
     "opts":["Mean","Median","Mode","Standard deviation"],
     "ah":h("Median")},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the overall default rate in the dataset?\n(Proportion of customers with is_defaulter=1)\n\nType rounded to 2 decimal places.",
     "ah":h(str(_defaulter_rate))},
    {"id":4,"type":"fill","marks":3,
     "text":"What is the median credit score of all customers?\n\nType the exact value.",
     "ah":h(str(_median_cs))},
    {"id":5,"type":"mcq","marks":3,
     "text":"In a box plot, the whiskers typically extend to:",
     "opts":[
         "Min and max values",
         "Q1 - 1.5*IQR and Q3 + 1.5*IQR",
         "Mean ± 2 standard deviations",
         "10th and 90th percentiles"
     ],
     "ah":h("Q1 - 1.5*IQR and Q3 + 1.5*IQR")},
    {"id":6,"type":"mcq","marks":3,
     "text":"What does a Pearson correlation of -0.85 between credit score and default flag indicate?",
     "opts":[
         "Higher credit score → higher default probability",
         "Strong negative relationship — higher credit score → lower default probability",
         "Weak negative relationship — almost no correlation",
         "No linear relationship exists"
     ],
     "ah":h("Strong negative relationship — higher credit score → lower default probability")},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nPrint the value counts of is_defaulter as a dictionary.\nFormat: {0: count, 1: count}",
     "preload":DF_SETUP,"exp":_p1e7},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nCompute the skewness of the income column.\nPrint rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_p1e8},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (400 rows).\n\nCheck for missing values across all columns.\nPrint total count of missing values in the entire dataset.",
     "preload":DF_SETUP,"exp":_p1e9},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 2 — EDA Workflow + Outlier Detection
# ════════════════════════════════════════════════════════════════
M3_PRACTICE_2 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"Which plot is best suited to visualize the distribution of a continuous variable like income?",
     "opts":["Bar chart","Scatter plot","Histogram","Pie chart"],
     "ah":h("Histogram")},
    {"id":2,"type":"mcq","marks":2,
     "text":"What does IQR stand for and what does it measure?",
     "opts":[
         "Inter-Quartile Range — spread of the middle 50% of data",
         "Inner-Quartile Range — distance between mean and median",
         "Integrated Quantile Ratio — measure of skewness",
         "None of the above"
     ],
     "ah":h("Inter-Quartile Range — spread of the middle 50% of data")},
    {"id":3,"type":"fill","marks":3,
     "text":"Which region has the highest default rate in the dataset?\n\nType the exact region name.",
     "ah":h(str(_highest_def_reg))},
    {"id":4,"type":"fill","marks":3,
     "text":"What is the IQR of loan_amount?\n\nIQR = Q3 - Q1\nType rounded to 2 decimal places.",
     "ah":h(str(_iqr_loan))},
    {"id":5,"type":"mcq","marks":3,
     "text":"When performing EDA, you notice a feature has 40% missing values. What is the best approach?",
     "opts":[
         "Always drop the column",
         "Always impute with mean",
         "Analyze if missingness is random or systematic, then decide",
         "Replace with 0"
     ],
     "ah":h("Analyze if missingness is random or systematic, then decide")},
    {"id":6,"type":"mcq","marks":3,
     "text":"A scatter plot between two variables shows a curved (non-linear) pattern. What does this mean for Pearson correlation?",
     "opts":[
         "Pearson correlation will capture the full relationship",
         "Pearson correlation may underestimate the true relationship",
         "Pearson correlation will be exactly 1",
         "The variables are independent"
     ],
     "ah":h("Pearson correlation may underestimate the true relationship")},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nFind the region with the highest average default rate.\nPrint only the region name.",
     "preload":DF_SETUP,"exp":_p2e7},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nCalculate the IQR of loan_amount.\nPrint rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_p2e8},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (400 rows).\n\nIdentify outliers in credit_score using the 2-standard-deviation rule.\n(Values more than 2 std from mean are outliers)\nPrint count of outliers.",
     "preload":DF_SETUP,"exp":_p2e9},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 3 — Correlation + Segmentation
# ════════════════════════════════════════════════════════════════
M3_PRACTICE_3 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What is the key difference between Pearson and Spearman correlation?",
     "opts":[
         "Pearson measures linear relationships; Spearman measures monotonic relationships",
         "Spearman requires normal distribution; Pearson does not",
         "Pearson works on categorical data; Spearman works on numerical data",
         "They are identical in all cases"
     ],
     "ah":h("Pearson measures linear relationships; Spearman measures monotonic relationships")},
    {"id":2,"type":"mcq","marks":2,
     "text":"In EDA, what does a bimodal histogram typically suggest?",
     "opts":[
         "Data has no outliers",
         "Data may consist of two distinct subpopulations",
         "Data is normally distributed",
         "Data has high variance"
     ],
     "ah":h("Data may consist of two distinct subpopulations")},
    {"id":3,"type":"fill","marks":3,
     "text":"What is the Pearson correlation between credit_score and is_defaulter?\n\nType rounded to 2 decimal places.",
     "ah":h(str(_corr_cs_def))},
    {"id":4,"type":"fill","marks":3,
     "text":"Which gender has the higher average loan_amount?\n\nType M or F.",
     "ah":h(str(_gender_loan_max))},
    {"id":5,"type":"mcq","marks":3,
     "text":"Which visualization is best for identifying the relationship between credit score and default status simultaneously across regions?",
     "opts":[
         "Pie chart",
         "Faceted box plot (one box plot per region)",
         "Line chart",
         "Single histogram"
     ],
     "ah":h("Faceted box plot (one box plot per region)")},
    {"id":6,"type":"mcq","marks":3,
     "text":"You bin credit scores into Low/Medium/High and find High segment has lowest default rate. What type of analysis is this?",
     "opts":[
         "Time series analysis",
         "Bivariate segmentation analysis",
         "Clustering",
         "Regression analysis"
     ],
     "ah":h("Bivariate segmentation analysis")},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nCompute Pearson correlation between credit_score and is_defaulter.\nPrint rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_p3e7},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\n\nWhich gender has the higher average loan_amount?\nPrint only M or F.",
     "preload":DF_SETUP,"exp":_p3e8},
    {"id":9,"type":"code","marks":5,
     "text":"df is preloaded (400 rows).\n\nBin credit_score into 3 segments:\n    bins: [0, 550, 700, 900]   labels: ['Low', 'Medium', 'High']\n\nWhich segment has the most customers?\nPrint only the segment name.",
     "preload":DF_SETUP,"exp":_p3e9},
]

# ════════════════════════════════════════════════════════════════
# FINAL ASSESSMENT — Primarily coding, all data visible
# ════════════════════════════════════════════════════════════════
M3_FINAL = [
    {"id":1,"type":"code","marks":3,
     "text":"df is preloaded (400 rows, 12 columns).\n\nCompute the mean income of all customers.\nPrint rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_fe1},
    {"id":2,"type":"code","marks":3,
     "text":"df is preloaded.\n\nWhat is the median credit score of all customers?\nPrint the exact value.",
     "preload":DF_SETUP,"exp":_fe2},
    {"id":3,"type":"code","marks":4,
     "text":"df is preloaded.\n\nCompute the default rate (mean of is_defaulter) for each region.\nPrint as a dict with values rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_fe3},
    {"id":4,"type":"code","marks":4,
     "text":"df is preloaded.\n\nUsing the IQR method (1.5*IQR rule), count how many loan_amount values\nare outliers (below Q1-1.5*IQR or above Q3+1.5*IQR).\nPrint the count.",
     "preload":DF_SETUP,"exp":_fe4},
    {"id":5,"type":"code","marks":5,
     "text":"df is preloaded.\n\nAmong these four columns:\n  income, credit_score, loan_amount, emi_amount\n\nFind which has the highest absolute Pearson correlation with is_defaulter.\nPrint only the column name.",
     "preload":DF_SETUP,"exp":_fe5},
    {"id":6,"type":"code","marks":4,
     "text":"df is preloaded.\n\nFind the loan_tenure value with the highest average default rate.\nPrint only the tenure value.",
     "preload":DF_SETUP,"exp":_fe6},
    {"id":7,"type":"code","marks":5,
     "text":"df is preloaded.\n\nCompute the difference in average credit_score between:\n  defaulters (is_defaulter=1) and non-defaulters (is_defaulter=0)\n\nPrint: defaulter_mean - non_defaulter_mean, rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_fe7},
    {"id":8,"type":"code","marks":5,
     "text":"df is preloaded.\n\nBin income into 3 equal-width bands: Low, Medium, High.\nFind which income band has the highest default rate.\nPrint only the band name.",
     "preload":DF_SETUP,"exp":_fe8},
    {"id":9,"type":"code","marks":4,
     "text":"df is preloaded.\n\nHow many customers have credit_score < 550 AND are NOT defaulters (is_defaulter=0)?\nPrint the count.",
     "preload":DF_SETUP,"exp":_fe9},
    {"id":10,"type":"code","marks":4,
     "text":"df is preloaded.\n\nCompute the standard deviation of emi_amount.\nPrint rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_fe10},
    {"id":11,"type":"code","marks":5,
     "text":"df is preloaded.\n\nFor each value of num_products (1 to 5),\ncompute the average default rate.\nPrint as a dict with values rounded to 2 decimal places.",
     "preload":DF_SETUP,"exp":_fe11},
    {"id":12,"type":"mcq","marks":3,
     "text":"In the dataset, income has a skewness of -0.03. What does this indicate?",
     "opts":[
         "Strong right skew — many high income outliers",
         "Strong left skew — many low income outliers",
         "Nearly symmetric distribution",
         "Bimodal distribution"
     ],
     "ah":h("Nearly symmetric distribution")},
    {"id":13,"type":"mcq","marks":4,
     "text":"Which visualization would best show how default rate varies across both region and gender simultaneously?",
     "opts":[
         "Single bar chart of default rate by region",
         "Grouped bar chart with region on x-axis and hue=gender",
         "Pie chart for each region",
         "Line plot of default rate over time"
     ],
     "ah":h("Grouped bar chart with region on x-axis and hue=gender")},
]

# ── MAPS ─────────────────────────────────────────────────────────
M3_PRACTICE_TESTS = {1: M3_PRACTICE_1, 2: M3_PRACTICE_2, 3: M3_PRACTICE_3}
M3_FINAL_TEST     = {1: M3_FINAL}
M3_DATASET        = _df.copy()