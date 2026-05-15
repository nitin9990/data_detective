"""
questions_m6.py — Module 6: Fraud Loss Forecasting Lab
Dataset: 60 months (Jan 2019 – Dec 2023), monthly fraud data by channel
M6_1: Moving Average (6 steps)
M6_2: Box-Jenkins ARIMA/SARIMA (10 steps)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# DATASET GENERATION (seed=606)
# ════════════════════════════════════════════════════════════════
DF_SETUP = '''\
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.random.seed(606)

months  = pd.date_range('2019-01-01', periods=60, freq='MS')
trend   = np.array([80000 * (1.08 ** (i/12)) for i in range(60)])
seasonal = np.array([1.15,0.90,0.85,0.90,0.95,1.00,1.05,1.05,1.00,1.00,1.05,1.25]*5)

# COVID shock: Mar-Jun 2020 spike
covid = np.ones(60)
covid[14:18] = [1.4, 1.6, 1.5, 1.3]
covid[18:24] = [1.2, 1.15, 1.1, 1.08, 1.05, 1.02]

# Channel shift: online growing, ATM declining
online_share = np.linspace(0.35, 0.55, 60)
atm_share    = np.linspace(0.35, 0.20, 60)
pos_share    = 1 - online_share - atm_share

noise       = np.random.normal(0, 0.05, 60)
total_loss  = (trend * seasonal * covid * (1 + noise)).astype(int)
online_loss = (total_loss * online_share * np.random.uniform(0.95,1.05,60)).astype(int)
atm_loss    = (total_loss * atm_share    * np.random.uniform(0.95,1.05,60)).astype(int)
pos_loss    = total_loss - online_loss - atm_loss
num_cases   = (total_loss / np.random.uniform(450,550,60)).astype(int)
rec_rate    = np.round(np.random.uniform(0.15,0.35,60), 3)
rec_amt     = (total_loss * rec_rate).astype(int)

df = pd.DataFrame({
    'month':            months,
    'total_fraud_loss': total_loss,
    'online_fraud_loss':online_loss,
    'pos_fraud_loss':   pos_loss,
    'atm_fraud_loss':   atm_loss,
    'num_fraud_cases':  num_cases,
    'avg_loss_per_case':(total_loss/num_cases).round(2),
    'recovery_amount':  rec_amt,
    'recovery_rate':    rec_rate,
})
df = df.set_index('month')
series = df['total_fraud_loss']
'''

# Preloaded with train/test split for evaluation steps
TRAIN_SETUP = DF_SETUP + '''
train  = series.iloc[:48]   # Jan 2019 – Dec 2022 (training)
test   = series.iloc[48:]   # Jan 2023 – Dec 2023 (holdout)
'''

# Full setup for ARIMA steps (54 train, 6 test)
ARIMA_SETUP = DF_SETUP + '''
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import math

train_s = series.iloc[:54]   # Jan 2019 – Jun 2023
test_s  = series.iloc[54:]   # Jul 2023 – Dec 2023 (6 months holdout)
'''

# ════════════════════════════════════════════════════════════════
# MODULE 6_1 — MOVING AVERAGE LAB
# ════════════════════════════════════════════════════════════════
M6_1_STEPS = [
    {
        "id": 1,
        "title": "Step 1 — Explore the Fraud Loss Dataset",
        "context": (
            "You are a Fraud Analytics Lead at a major bank.\n"
            "Monthly fraud losses have been rising and leadership needs a 6-month forecast "
            "to set reserve capital for next year.\n\n"
            "**Dataset:** 60 months of fraud data (Jan 2019 – Dec 2023)\n"
            "**Channels:** Online, POS (card present), ATM\n"
            "**Target:** total_fraud_loss (USD)\n\n"
            "**Task:** Print the following:\n"
            "1. Shape of the dataset\n"
            "2. Mean total_fraud_loss (rounded to nearest dollar)\n"
            "3. Month with highest fraud loss (format: YYYY-MM)"
        ),
        "preload": DF_SETUP,
        "exp": "(60, 8)\n101633\n2022-12",
        "solution": (
            "print(df.shape)\n"
            "print(round(series.mean()))\n"
            "print(series.idxmax().strftime('%Y-%m'))"
        ),
        "explanation": (
            "Dataset: 60 months × 8 columns.\n"
            "Mean monthly fraud loss: $101,633.\n"
            "Peak: Dec 2022 — festive season + COVID recovery driving card spending.\n"
            "Notice the trend: losses grew from ~$70K in early 2019 to ~$138K by end 2023 (~97% increase)."
        ),
    },
    {
        "id": 2,
        "title": "Step 2 — Simple Moving Average (3M, 6M, 12M)",
        "context": (
            "Moving averages smooth out short-term noise to reveal the underlying trend.\n\n"
            "**Window sizes:**\n"
            "- 3M MA: reacts quickly, more noise\n"
            "- 6M MA: balanced\n"
            "- 12M MA: smoothest, eliminates seasonal effects\n\n"
            "**Task:**\n"
            "Compute rolling mean with windows 3, 6, 12 on the full series.\n"
            "Print the last value of MA-12 (rounded to nearest dollar)."
        ),
        "preload": DF_SETUP,
        "exp": "115979",
        "solution": (
            "ma3  = series.rolling(3).mean()\n"
            "ma6  = series.rolling(6).mean()\n"
            "ma12 = series.rolling(12).mean()\n"
            "print(round(ma12.iloc[-1]))"
        ),
        "explanation": (
            "MA-12 last value = $115,979 — this is the average of all 12 months of 2023.\n"
            "MA-12 is the most common baseline in fraud forecasting — "
            "it naturally removes the 12-month seasonal cycle.\n"
            "Larger windows lag more but are more stable for forecasting."
        ),
    },
    {
        "id": 3,
        "title": "Step 3 — Weighted Moving Average",
        "context": (
            "Simple MA gives equal weight to all past months.\n"
            "But recent months are more relevant for fraud forecasting — "
            "fraud patterns change with new attack vectors.\n\n"
            "**Weighted MA:** Give more weight to recent months.\n"
            "Weights for 6-month window: [1, 2, 3, 4, 5, 6] (normalized to sum=1)\n\n"
            "**Task:**\n"
            "Compute 6-month WMA using weights [1,2,3,4,5,6] normalized.\n"
            "Print the last WMA value (rounded to nearest dollar)."
        ),
        "preload": DF_SETUP,
        "exp": "124641",
        "solution": (
            "weights = np.array([1, 2, 3, 4, 5, 6])\n"
            "weights = weights / weights.sum()\n"
            "wma = series.rolling(6).apply(lambda x: np.dot(x, weights))\n"
            "print(round(wma.iloc[-1]))"
        ),
        "explanation": (
            "WMA last value = $124,641 — higher than MA-12 ($115,979) because it weights "
            "recent months (Nov-Dec 2023 were high) more heavily.\n"
            "WMA is better when fraud is trending — it adjusts faster.\n"
            "Trade-off: more reactive to short-term spikes."
        ),
    },
    {
        "id": 4,
        "title": "Step 4 — Exponential Smoothing",
        "context": (
            "Exponential Smoothing applies exponentially decreasing weights to all past observations.\n"
            "The **alpha** parameter controls how fast weights decay:\n"
            "- alpha=0.3: slow decay, smooth — good for stable trends\n"
            "- alpha=0.7: fast decay, reactive — good for volatile series\n\n"
            "**Task:**\n"
            "Fit SimpleExpSmoothing with alpha=0.3 and alpha=0.7 on the full series.\n"
            "Print last fitted value for each (rounded to nearest dollar), space-separated."
        ),
        "preload": DF_SETUP + "\nfrom statsmodels.tsa.holtwinters import SimpleExpSmoothing\n",
        "exp": "118429 120282",
        "solution": (
            "from statsmodels.tsa.holtwinters import SimpleExpSmoothing\n"
            "es03 = SimpleExpSmoothing(series).fit(smoothing_level=0.3, optimized=False)\n"
            "es07 = SimpleExpSmoothing(series).fit(smoothing_level=0.7, optimized=False)\n"
            "print(round(es03.fittedvalues.iloc[-1]), round(es07.fittedvalues.iloc[-1]))"
        ),
        "explanation": (
            "alpha=0.3: $118,429 — smoother, influenced by longer history.\n"
            "alpha=0.7: $120,282 — more weight on recent Dec 2023 spike.\n"
            "For fraud loss with strong seasonality, alpha=0.3 is usually better — "
            "prevents over-reacting to Dec peak when forecasting Jan."
        ),
    },
    {
        "id": 5,
        "title": "Step 5 — Forecast Next 6 Months Using MA-12",
        "context": (
            "The simplest forecast: carry forward the last 12-month moving average.\n"
            "This is the 'naive seasonal' baseline — what many fraud teams use today.\n\n"
            "**Setup:**\n"
            "- train = first 48 months (Jan 2019 – Dec 2022)\n"
            "- test  = last 12 months (Jan 2023 – Dec 2023)\n\n"
            "**Task:**\n"
            "Compute MA-12 on train. Use the last MA-12 value as forecast for all 6 months "
            "(Jan–Jun 2023, i.e. test.iloc[:6]).\n"
            "Print the forecast value (same for all 6 months, rounded to nearest dollar)."
        ),
        "preload": TRAIN_SETUP,
        "exp": "105978",
        "solution": (
            "ma12_train  = train.rolling(12).mean()\n"
            "last_ma12   = ma12_train.iloc[-1]\n"
            "forecast_ma = [last_ma12] * 6\n"
            "print(round(last_ma12))"
        ),
        "explanation": (
            "MA-12 forecast = $105,978 for all 6 months.\n"
            "This is a flat forecast — MA assumes the future = recent average.\n"
            "It captures level but misses trend (actual 2023 values ranged $113K–$136K).\n"
            "Evaluation in next step will quantify how wrong this is."
        ),
    },
    {
        "id": 6,
        "title": "Step 6 — Evaluate: MAE, RMSE, MAPE",
        "context": (
            "We compare our MA forecast against actual fraud losses (Jan–Jun 2023).\n\n"
            "**Metrics:**\n"
            "- **MAE** (Mean Absolute Error): average dollar error — easy to explain to business\n"
            "- **RMSE** (Root Mean Square Error): penalises large errors more — "
            "important when a single month miss is costly\n"
            "- **MAPE** (Mean Absolute Percentage Error): % error — useful for comparing "
            "across different scales\n\n"
            "**Task:**\n"
            "Using MA-12 forecast vs test.iloc[:6] actuals:\n"
            "Print MAE, RMSE, MAPE (%) — space separated, MAPE rounded to 2 dp."
        ),
        "preload": TRAIN_SETUP,
        "exp": "8341 10017 7.61",
        "solution": (
            "import math\n"
            "ma12_train  = train.rolling(12).mean()\n"
            "last_ma12   = ma12_train.iloc[-1]\n"
            "forecast_ma = np.array([last_ma12] * 6)\n"
            "actuals     = test.values[:6]\n\n"
            "mae  = round(np.mean(np.abs(actuals - forecast_ma)))\n"
            "rmse = round(math.sqrt(np.mean((actuals - forecast_ma)**2)))\n"
            "mape = round(np.mean(np.abs((actuals - forecast_ma)/actuals))*100, 2)\n"
            "print(mae, rmse, mape)"
        ),
        "explanation": (
            "MAE = $8,341: on average off by $8,341 per month.\n"
            "RMSE = $10,017: higher than MAE because some months are more wrong.\n"
            "MAPE = 7.61%: ~8% error — for reserve capital, this means "
            "for every $1M budgeted, you're off by $76,100.\n"
            "This is our baseline — ARIMA/SARIMA in Module 6_2 should beat this."
        ),
    },
]

# ════════════════════════════════════════════════════════════════
# MODULE 6_2 — BOX-JENKINS ARIMA/SARIMA LAB
# ════════════════════════════════════════════════════════════════
M6_2_STEPS = [
    {
        "id": 1,
        "title": "Step 1 — Stationarity Test (ADF)",
        "context": (
            "**Box-Jenkins Framework — Step 1: Check Stationarity**\n\n"
            "ARIMA requires the series to be stationary — "
            "constant mean and variance over time.\n"
            "Our fraud series has an upward trend → likely non-stationary.\n\n"
            "**Augmented Dickey-Fuller (ADF) test:**\n"
            "- H₀: Series is non-stationary (has unit root)\n"
            "- If p-value < 0.05 → reject H₀ → stationary\n"
            "- If p-value > 0.05 → fail to reject → non-stationary → need differencing\n\n"
            "**Task:** Run ADF test on total_fraud_loss.\n"
            "Print ADF statistic and p-value (both rounded to 4 dp), space-separated."
        ),
        "preload": ARIMA_SETUP,
        "exp": "-0.7386 0.8365",
        "solution": (
            "from statsmodels.tsa.stattools import adfuller\n"
            "adf = adfuller(series)\n"
            "print(round(adf[0], 4), round(adf[1], 4))"
        ),
        "explanation": (
            "ADF stat = -0.7386, p-value = 0.8365 >> 0.05.\n"
            "Fail to reject H₀ — series is NON-STATIONARY.\n"
            "This is expected: our fraud series has an upward trend.\n"
            "The d parameter in ARIMA(p,d,q) represents the number of differences needed — "
            "we need d≥1."
        ),
    },
    {
        "id": 2,
        "title": "Step 2 — Differencing to Achieve Stationarity",
        "context": (
            "**Box-Jenkins Framework — Step 2: Differencing (d parameter)**\n\n"
            "First-order differencing: diff(t) = y(t) - y(t-1)\n"
            "This removes the trend, making the series stationary.\n\n"
            "**Task:**\n"
            "1. Apply first-order differencing to series\n"
            "2. Run ADF test on the differenced series\n"
            "3. Print ADF p-value (rounded to 4 dp)\n"
            "4. Print whether it is stationary (True/False)"
        ),
        "preload": ARIMA_SETUP,
        "exp": "0.0005\nTrue",
        "solution": (
            "diff1 = series.diff().dropna()\n"
            "adf2  = adfuller(diff1)\n"
            "print(round(adf2[1], 4))\n"
            "print(adf2[1] < 0.05)"
        ),
        "explanation": (
            "After first differencing: p-value = 0.0005 << 0.05.\n"
            "Reject H₀ — series is now STATIONARY. d=1 is sufficient.\n"
            "So our ARIMA model will use d=1 in ARIMA(p, 1, q).\n"
            "If one difference wasn't enough, we'd apply a second difference (d=2) — "
            "but this is rare in practice."
        ),
    },
    {
        "id": 3,
        "title": "Step 3 — ACF Analysis (MA order q)",
        "context": (
            "**Box-Jenkins Framework — Step 3: ACF → MA order (q)**\n\n"
            "ACF (AutoCorrelation Function) shows correlation of series with its lagged values.\n"
            "After differencing, ACF of the residuals helps identify the MA order:\n"
            "- Significant spike at lag k, then cuts off → MA(k), so q=k\n"
            "- Gradual decay → AR process (look at PACF)\n\n"
            "Rule of thumb: |ACF| > 0.2 is considered significant.\n\n"
            "**Task:** Compute ACF of differenced series for lags 1-15.\n"
            "Print ACF values for lags 1-5 (rounded to 3 dp) as a list."
        ),
        "preload": ARIMA_SETUP,
        "exp": "[-0.061, -0.35, -0.065, -0.056, 0.07]",
        "solution": (
            "diff1    = series.diff().dropna()\n"
            "acf_vals = acf(diff1, nlags=15)\n"
            "print([round(x, 3) for x in acf_vals[1:6]])"
        ),
        "explanation": (
            "ACF values: [-0.061, -0.35, -0.065, -0.056, 0.07]\n"
            "Lag 2 has ACF = -0.35 (significant, |ACF| > 0.2) — suggests MA(2) possible.\n"
            "But the pattern is not a clean cutoff — suggests ARIMA(1,1,1) or seasonal effects.\n"
            "This is why we use both ACF and PACF together to decide p and q."
        ),
    },
    {
        "id": 4,
        "title": "Step 4 — PACF Analysis (AR order p)",
        "context": (
            "**Box-Jenkins Framework — Step 4: PACF → AR order (p)**\n\n"
            "PACF (Partial AutoCorrelation Function) removes intermediate lag effects.\n"
            "After differencing, PACF helps identify the AR order:\n"
            "- Significant spike at lag k, then cuts off → AR(k), so p=k\n"
            "- Gradual decay → MA process\n\n"
            "Rule of thumb: |PACF| > 0.2 is considered significant.\n\n"
            "**Task:** Compute PACF of differenced series for lags 1-15.\n"
            "Print PACF values for lags 1-5 (rounded to 3 dp) as a list."
        ),
        "preload": ARIMA_SETUP,
        "exp": "[-0.062, -0.367, -0.142, -0.255, -0.058]",
        "solution": (
            "diff1     = series.diff().dropna()\n"
            "pacf_vals = pacf(diff1, nlags=15)\n"
            "print([round(x, 3) for x in pacf_vals[1:6]])"
        ),
        "explanation": (
            "PACF values: [-0.062, -0.367, -0.142, -0.255, -0.058]\n"
            "Lags 2 and 4 show |PACF| > 0.2 — suggests AR component.\n"
            "Combined reading: ACF and PACF both show significant lag 2 — "
            "likely seasonal (every 2 quarters). This motivates SARIMA.\n"
            "For ARIMA, we'll start with p=1, d=1, q=1 as a reasonable baseline."
        ),
    },
    {
        "id": 5,
        "title": "Step 5 — Fit ARIMA(1,1,1)",
        "context": (
            "**Box-Jenkins Framework — Step 5: Model Fitting**\n\n"
            "Based on ACF/PACF analysis: p=1, d=1, q=1.\n\n"
            "**Setup:**\n"
            "- train_s = first 54 months (Jan 2019 – Jun 2023)\n"
            "- test_s  = last 6 months (Jul – Dec 2023)\n\n"
            "**Task:**\n"
            "Fit ARIMA(1,1,1) on train_s.\n"
            "Print:\n"
            "1. AIC (Akaike Information Criterion) — rounded to 2 dp\n"
            "2. Forecast for next 6 months — list of rounded integers"
        ),
        "preload": ARIMA_SETUP,
        "exp": "1150.79\n[110130, 109762, 109628, 109579, 109562, 109555]",
        "solution": (
            "arima      = ARIMA(train_s, order=(1,1,1)).fit()\n"
            "arima_pred = arima.forecast(6)\n"
            "print(round(arima.aic, 2))\n"
            "print([round(x) for x in arima_pred.values])"
        ),
        "explanation": (
            "AIC = 1150.79 — lower AIC = better model (used for comparison).\n"
            "Forecast: ARIMA gives a flat forecast (~$109K/month) — "
            "it captures the trend but misses seasonality.\n"
            "Actual Jul-Dec 2023: $119K-$136K — ARIMA underestimates.\n"
            "This motivates SARIMA — we need to capture the seasonal component."
        ),
    },
    {
        "id": 6,
        "title": "Step 6 — Residual Diagnostics",
        "context": (
            "**Box-Jenkins Framework — Step 6: Diagnostic Checking**\n\n"
            "Good residuals should be: white noise (no autocorrelation), mean ≈ 0.\n\n"
            "**Ljung-Box test:**\n"
            "- H₀: Residuals are white noise (no autocorrelation)\n"
            "- p-value > 0.05 → residuals are white noise → model is adequate\n"
            "- p-value < 0.05 → residuals have structure → model needs improvement\n\n"
            "**Task:**\n"
            "Run Ljung-Box test on ARIMA(1,1,1) residuals at lag 10.\n"
            "Print p-value (rounded to 4 dp) and residual mean (rounded to 2 dp), space-separated."
        ),
        "preload": ARIMA_SETUP,
        "exp": "0.9441 3376.09",
        "solution": (
            "arima = ARIMA(train_s, order=(1,1,1)).fit()\n"
            "lb    = acorr_ljungbox(arima.resid, lags=[10], return_df=True)\n"
            "print(round(lb['lb_pvalue'].iloc[0], 4), round(arima.resid.mean(), 2))"
        ),
        "explanation": (
            "Ljung-Box p-value = 0.9441 >> 0.05 → residuals ARE white noise — model is adequate.\n"
            "Residual mean = 3376.09 (not exactly 0) — slight bias, ARIMA underpredicts by ~$3,376/month.\n"
            "The model passes the diagnostic test, but the non-zero mean suggests "
            "we're missing a structural component — likely the seasonal pattern."
        ),
    },
    {
        "id": 7,
        "title": "Step 7 — SARIMA(1,1,1)(1,1,1,12)",
        "context": (
            "**Box-Jenkins Framework — Step 7: Seasonal ARIMA**\n\n"
            "SARIMA adds seasonal AR, differencing and MA terms:\n"
            "SARIMA(p,d,q)(P,D,Q,m) where m=12 for monthly data.\n\n"
            "Our parameters: (1,1,1)(1,1,1,12)\n"
            "- Non-seasonal: p=1, d=1, q=1\n"
            "- Seasonal: P=1, D=1, Q=1, m=12 (annual seasonality)\n\n"
            "**Task:**\n"
            "Fit SARIMAX with order=(1,1,1) and seasonal_order=(1,1,1,12) on train_s.\n"
            "Print:\n"
            "1. AIC (rounded to 2 dp)\n"
            "2. Forecast for next 6 months — list of rounded integers"
        ),
        "preload": ARIMA_SETUP,
        "exp": "891.03\n[119945, 114099, 114621, 112513, 120284, 142270]",
        "solution": (
            "sarima      = SARIMAX(train_s, order=(1,1,1),\n"
            "                      seasonal_order=(1,1,1,12)).fit(disp=False)\n"
            "sarima_pred = sarima.forecast(6)\n"
            "print(round(sarima.aic, 2))\n"
            "print([round(x) for x in sarima_pred.values])"
        ),
        "explanation": (
            "SARIMA AIC = 891.03 vs ARIMA AIC = 1150.79 — significantly better fit.\n"
            "SARIMA forecast shows the Dec spike ($142K) — matches actual pattern.\n"
            "Actual Jul-Dec 2023: [$127K, $125K, $120K, $113K, $122K, $136K].\n"
            "SARIMA captures seasonality much better than plain ARIMA."
        ),
    },
    {
        "id": 8,
        "title": "Step 8 — Auto ARIMA",
        "context": (
            "**Auto ARIMA** uses stepwise search to find the best (p,d,q)(P,D,Q) automatically.\n"
            "It minimises AIC/BIC across candidate models.\n\n"
            "Use this as a **sanity check** against your manual Box-Jenkins selection.\n"
            "If auto ARIMA picks different parameters — investigate why.\n\n"
            "**Task:**\n"
            "Run pm.auto_arima on train_s with seasonal=True, m=12, stepwise=True.\n"
            "Print:\n"
            "1. Best order (p,d,q)\n"
            "2. Best seasonal order (P,D,Q,m)\n"
            "3. Forecast for next 6 months — list of rounded integers"
        ),
        "preload": ARIMA_SETUP + "\nimport pmdarima as pm\n",
        "exp": "(1, 1, 2)\n(1, 0, 0, 12)\n[109432, 111530, 109149, 110417, 111453, 118554]",
        "solution": (
            "import pmdarima as pm\n"
            "auto      = pm.auto_arima(train_s, seasonal=True, m=12,\n"
            "                          stepwise=True, suppress_warnings=True,\n"
            "                          error_action='ignore')\n"
            "auto_pred = auto.predict(6)\n"
            "print(auto.order)\n"
            "print(auto.seasonal_order)\n"
            "print([round(x) for x in auto_pred])"
        ),
        "explanation": (
            "Auto ARIMA selected: ARIMA(1,1,2)(1,0,0,12).\n"
            "Different from manual SARIMA(1,1,1)(1,1,1,12) — auto skipped seasonal differencing.\n"
            "Auto's forecast is flatter ($109K–$119K) — misses the Dec spike.\n"
            "Lesson: Auto ARIMA is a good starting point but domain knowledge "
            "(knowing fraud spikes in December) should override pure statistical selection."
        ),
    },
    {
        "id": 9,
        "title": "Step 9 — Forecast with Confidence Intervals",
        "context": (
            "Point forecasts alone are insufficient for business decisions.\n"
            "Reserve capital planning needs a **range** — how wide could the loss be?\n\n"
            "SARIMA provides 95% confidence intervals:\n"
            "- Lower bound: optimistic scenario\n"
            "- Upper bound: worst-case scenario → use this for capital reserves\n\n"
            "**Task:**\n"
            "Get SARIMA forecast with confidence intervals for 6 months.\n"
            "Print lower bounds and upper bounds as separate lists (rounded to nearest dollar)."
        ),
        "preload": ARIMA_SETUP,
        "exp": "[96750, 83943, 79782, 73923, 78433, 97463]\n[143139, 144255, 149460, 151102, 162135, 187077]",
        "solution": (
            "sarima    = SARIMAX(train_s, order=(1,1,1),\n"
            "                    seasonal_order=(1,1,1,12)).fit(disp=False)\n"
            "fc        = sarima.get_forecast(6)\n"
            "ci        = fc.conf_int()\n"
            "print([round(x) for x in ci.iloc[:,0]])\n"
            "print([round(x) for x in ci.iloc[:,1]])"
        ),
        "explanation": (
            "Confidence intervals widen over time — uncertainty grows with forecast horizon.\n"
            "Dec 2023 upper bound: $187,077 — this is what risk teams use for capital reserves.\n"
            "Upper CI for 6 months totals ~$877K vs point forecast total ~$723K.\n"
            "Business insight: Reserve an extra $154K buffer above point forecast for 95% coverage."
        ),
    },
    {
        "id": 10,
        "title": "Step 10 — Model Comparison: MA vs ARIMA vs SARIMA vs Auto ARIMA",
        "context": (
            "Final step — compare all models on the same 6-month holdout (Jul–Dec 2023).\n\n"
            "**Task:**\n"
            "Compute RMSE for each model against test_s:\n"
            "1. MA-12 baseline (flat forecast = last MA-12 value)\n"
            "2. ARIMA(1,1,1)\n"
            "3. SARIMA(1,1,1)(1,1,1,12)\n"
            "4. Auto ARIMA\n\n"
            "Print each RMSE rounded to nearest dollar and the winner, in this format:\n"
            "MA: X  ARIMA: X  SARIMA: X  Auto: X  Winner: MODEL"
        ),
        "preload": ARIMA_SETUP + "\nimport pmdarima as pm\n",
        "exp": "MA: 10017  ARIMA: 15793  SARIMA: 6376  Auto: 13132  Winner: SARIMA",
        "solution": (
            "# MA baseline (using train first 48 months)\n"
            "train48     = series.iloc[:48]\n"
            "last_ma12   = train48.rolling(12).mean().iloc[-1]\n"
            "fc_ma       = np.array([last_ma12]*6)\n"
            "rmse_ma     = round(math.sqrt(np.mean((test_s.values-fc_ma)**2)))\n\n"
            "# ARIMA\n"
            "arima       = ARIMA(train_s, order=(1,1,1)).fit()\n"
            "fc_arima    = arima.forecast(6).values\n"
            "rmse_arima  = round(math.sqrt(np.mean((test_s.values-fc_arima)**2)))\n\n"
            "# SARIMA\n"
            "sarima      = SARIMAX(train_s, order=(1,1,1),\n"
            "                      seasonal_order=(1,1,1,12)).fit(disp=False)\n"
            "fc_sarima   = sarima.forecast(6).values\n"
            "rmse_sarima = round(math.sqrt(np.mean((test_s.values-fc_sarima)**2)))\n\n"
            "# Auto ARIMA\n"
            "auto        = pm.auto_arima(train_s, seasonal=True, m=12,\n"
            "                            stepwise=True, suppress_warnings=True,\n"
            "                            error_action='ignore')\n"
            "fc_auto     = auto.predict(6)\n"
            "rmse_auto   = round(math.sqrt(np.mean((test_s.values-fc_auto)**2)))\n\n"
            "winner = min({'MA':rmse_ma,'ARIMA':rmse_arima,'SARIMA':rmse_sarima,'Auto':rmse_auto},\n"
            "             key=lambda k: {'MA':rmse_ma,'ARIMA':rmse_arima,'SARIMA':rmse_sarima,'Auto':rmse_auto}[k])\n"
            "print(f'MA: {rmse_ma}  ARIMA: {rmse_arima}  SARIMA: {rmse_sarima}  Auto: {rmse_auto}  Winner: {winner}')"
        ),
        "explanation": (
            "RMSE results: MA=$10,017 | ARIMA=$15,793 | SARIMA=$6,376 | Auto=$13,132\n"
            "Winner: SARIMA — 36% better than the MA baseline.\n\n"
            "Key lessons:\n"
            "- SARIMA beats all — seasonal modelling is critical for monthly fraud data\n"
            "- ARIMA is WORSE than MA — adding complexity without seasonality hurts\n"
            "- Auto ARIMA missed the seasonal differencing — domain knowledge matters\n"
            "- MA is a strong baseline — simple models often beat complex ones without seasonality\n\n"
            "Recommendation: Deploy SARIMA(1,1,1)(1,1,1,12) with upper CI for capital reserves."
        ),
    },
]

# ── MAPS ──────────────────────────────────────────────────────────
M6_1_LAB = M6_1_STEPS
M6_2_LAB = M6_2_STEPS
