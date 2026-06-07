import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

st.set_page_config(
    page_title="Capstone 2 — Initial Credit Line Assignment",
    page_icon="💳", layout="wide"
)

def _sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def log_download(email, employee_id):
    try:
        _sb().table("attempts").insert({
            "email":       email.lower().strip(),
            "level":       "capstone2_creditline",
            "test_id":     1,
            "attempt_num": 1,
            "max_score":   100,
            "score":       0,
            "employee_id": employee_id.strip(),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "valid":       True,
        }).execute()
    except Exception as e:
        st.warning(f"Logging failed: {e}")

@st.cache_data
def generate_dataset():
    import warnings; warnings.filterwarnings('ignore')
    np.random.seed(303); n = 30000

    application_channel    = np.random.choice(['Branch','DSA','Online','Pre-approved'],n,p=[0.20,0.35,0.30,0.15])
    product_applied        = np.random.choice(['Classic','Gold','Platinum','Signature'],n,p=[0.30,0.35,0.25,0.10])
    requested_limit        = np.random.choice([25000,50000,75000,100000,150000,200000,300000,500000],n,
                                               p=[0.10,0.20,0.20,0.20,0.15,0.08,0.05,0.02])
    city_tier              = np.random.choice(['Tier1','Tier2','Tier3'],n,p=[0.35,0.35,0.30])
    state                  = np.random.choice(['MH','DL','KA','TN','UP','GJ','RJ','WB','AP','Others'],n)
    residence_type         = np.random.choice(['Owned','Rented','Family-owned','Company-provided'],n,p=[0.35,0.40,0.15,0.10])
    residence_stability    = np.random.randint(0,120,n)
    existing_bank_rel      = np.random.choice([0,1],n,p=[0.40,0.60])
    num_existing_products  = np.where(existing_bank_rel, np.random.randint(1,5,n), 0)
    preferred_lang         = np.random.choice(['English','Hindi','Regional'],n,p=[0.30,0.45,0.25])

    employer_category      = np.random.choice(['PSU','Large-Private','SME','Govt','Self-Employed','Gig'],n,p=[0.12,0.38,0.15,0.12,0.18,0.05])
    employment_type        = np.random.choice(['Salaried','Self-Employed','Business-Owner','Professional'],n,p=[0.55,0.20,0.15,0.10])
    years_in_current_job   = np.round(np.random.exponential(3,n).clip(0,30),1)
    total_work_exp_years   = years_in_current_job + np.round(np.random.exponential(2,n).clip(0,20),1)
    gross_annual_income    = np.random.randint(150000,5000000,n)
    net_monthly_income     = (gross_annual_income/12*np.random.uniform(0.65,0.80,n)).astype(int)
    income_proof_type      = np.random.choice(['ITR','Form16','Bank-Statement','CA-Certificate','None'],n,p=[0.30,0.35,0.15,0.10,0.10])
    income_verified_flag   = np.random.choice([0,1],n,p=[0.15,0.85])
    salary_account_flag    = np.random.choice([0,1],n,p=[0.35,0.65])
    avg_monthly_credit     = np.where(salary_account_flag,(net_monthly_income*np.random.uniform(0.90,1.10,n)).astype(int),np.random.randint(0,50000,n))
    income_stability_score = np.round(np.random.beta(4,2,n)*100,1)
    employer_stability_score=np.round(np.random.beta(3,2,n)*100,1)

    bureau_score           = np.random.randint(300,850,n)
    bureau_score_vintage   = np.random.randint(6,60,n)
    num_active_loans       = np.random.randint(0,8,n)
    num_active_cards       = np.random.randint(0,5,n)
    num_closed_loans       = np.random.randint(0,10,n)
    total_active_balance   = np.random.randint(0,5000000,n)
    total_credit_limit_bureau=np.random.randint(0,2000000,n)
    bureau_utilization     = np.round(np.clip(total_active_balance/(total_credit_limit_bureau+1),0,1.5),3)
    credit_age_oldest      = np.random.randint(6,240,n)
    credit_age_newest      = np.random.randint(1,36,n)
    credit_age_avg         = (credit_age_oldest+credit_age_newest)//2
    num_inquiries_1m       = np.random.randint(0,5,n)
    num_inquiries_3m       = num_inquiries_1m+np.random.randint(0,5,n)
    num_inquiries_6m       = num_inquiries_3m+np.random.randint(0,5,n)
    dpd_30_ever            = np.random.choice([0,1],n,p=[0.65,0.35])
    dpd_60_ever            = np.random.choice([0,1],n,p=[0.80,0.20])
    dpd_90_ever            = np.random.choice([0,1],n,p=[0.88,0.12])
    written_off_ever       = np.random.choice([0,1],n,p=[0.94,0.06])
    settled_ever           = np.random.choice([0,1],n,p=[0.90,0.10])
    num_delinquent_now     = np.random.randint(0,3,n)

    home_loan_flag         = np.random.choice([0,1],n,p=[0.65,0.35])
    home_loan_emi          = np.where(home_loan_flag,np.random.randint(5000,80000,n),0)
    personal_loan_flag     = np.random.choice([0,1],n,p=[0.60,0.40])
    personal_loan_emi      = np.where(personal_loan_flag,np.random.randint(2000,30000,n),0)
    auto_loan_flag         = np.random.choice([0,1],n,p=[0.70,0.30])
    auto_loan_emi          = np.where(auto_loan_flag,np.random.randint(3000,25000,n),0)
    total_existing_emi     = home_loan_emi+personal_loan_emi+auto_loan_emi
    foir                   = np.round(total_existing_emi/(net_monthly_income+1),3).clip(0,2)
    num_credit_cards_existing=np.random.randint(0,5,n)
    total_card_limit_existing=np.where(num_credit_cards_existing>0,np.random.randint(0,500000,n),0)

    banking_vintage_months = np.where(existing_bank_rel,np.random.randint(6,120,n),0)
    avg_monthly_balance    = np.where(existing_bank_rel,np.random.randint(1000,500000,n),0)
    min_monthly_balance    = (avg_monthly_balance*np.random.uniform(0.3,0.8,n)).astype(int)
    balance_stability      = np.round(np.where(existing_bank_rel,np.random.beta(3,2,n),0),3)
    fixed_deposit_flag     = np.random.choice([0,1],n,p=[0.65,0.35])
    fd_amount              = np.where(fixed_deposit_flag,np.random.randint(10000,1000000,n),0)
    investment_flag        = np.random.choice([0,1],n,p=[0.75,0.25])
    premium_banking_flag   = np.random.choice([0,1],n,p=[0.78,0.22])

    internal_app_score     = np.round(np.random.uniform(0,1000,n),1)
    bureau_score_segment   = pd.cut(bureau_score,bins=[0,550,650,720,780,850],
                                     labels=['Very Poor','Poor','Fair','Good','Excellent']).astype(str)
    debt_to_income_ratio   = np.round(total_active_balance/(gross_annual_income+1),3).clip(0,5)
    repayment_capacity     = np.round((net_monthly_income-total_existing_emi)/(net_monthly_income+1),3).clip(0,1)
    leverage_ratio         = np.round(total_active_balance/(gross_annual_income+1),3).clip(0,10)
    credit_to_income       = np.round(total_credit_limit_bureau/(gross_annual_income+1),3).clip(0,10)
    payment_track_score    = np.round(np.where(dpd_30_ever==0,np.random.beta(5,2,n),np.random.beta(2,5,n)),3)
    relationship_score     = np.round(np.where(existing_bank_rel,np.random.beta(4,2,n),np.random.beta(2,4,n)),3)
    fraud_risk_score       = np.round(np.random.beta(1,10,n)*100,1)
    policy_override_flag   = np.random.choice([0,1],n,p=[0.92,0.08])

    approval_logodds = (-1.0+0.004*(bureau_score-300)-2.0*written_off_ever-1.5*dpd_90_ever-1.0*dpd_60_ever-0.5*dpd_30_ever-1.5*foir.clip(0,1)+1.0*income_verified_flag+0.5*salary_account_flag-1.0*(num_inquiries_3m/10)-1.0*num_delinquent_now+0.5*existing_bank_rel+np.random.normal(0,0.5,n))
    approved = (np.random.uniform(0,1,n)<1/(1+np.exp(-approval_logodds))).astype(int)

    base_limit = (0.30*(gross_annual_income/12)+0.20*(gross_annual_income/12)*((bureau_score-300)/550)+0.15*(gross_annual_income/12)*income_verified_flag-0.10*(gross_annual_income/12)*foir.clip(0,1)+0.10*(gross_annual_income/12)*existing_bank_rel+0.05*(gross_annual_income/12)*fixed_deposit_flag+np.random.normal(0,gross_annual_income/24,n))
    credit_limit_assigned = np.round(base_limit/5000)*5000
    credit_limit_assigned = np.clip(credit_limit_assigned,10000,500000).astype(int)
    credit_limit_assigned = np.where(approved, credit_limit_assigned, 0)

    return pd.DataFrame({
        'application_id':range(300001,330001),
        'application_channel':application_channel,'product_applied':product_applied,
        'requested_limit':requested_limit,'city_tier':city_tier,'state':state,
        'residence_type':residence_type,'residence_stability':residence_stability,
        'existing_bank_rel':existing_bank_rel,'num_existing_products':num_existing_products,
        'preferred_lang':preferred_lang,
        'employer_category':employer_category,'employment_type':employment_type,
        'years_in_current_job':years_in_current_job,'total_work_exp_years':total_work_exp_years,
        'gross_annual_income':gross_annual_income,'net_monthly_income':net_monthly_income,
        'income_proof_type':income_proof_type,'income_verified_flag':income_verified_flag,
        'salary_account_flag':salary_account_flag,'avg_monthly_credit':avg_monthly_credit,
        'income_stability_score':income_stability_score,
        'employer_stability_score':employer_stability_score,
        'bureau_score':bureau_score,'bureau_score_vintage':bureau_score_vintage,
        'num_active_loans':num_active_loans,'num_active_cards':num_active_cards,
        'num_closed_loans':num_closed_loans,'total_active_balance':total_active_balance,
        'total_credit_limit_bureau':total_credit_limit_bureau,
        'bureau_utilization':bureau_utilization,'credit_age_oldest':credit_age_oldest,
        'credit_age_newest':credit_age_newest,'credit_age_avg':credit_age_avg,
        'num_inquiries_1m':num_inquiries_1m,'num_inquiries_3m':num_inquiries_3m,
        'num_inquiries_6m':num_inquiries_6m,'dpd_30_ever':dpd_30_ever,
        'dpd_60_ever':dpd_60_ever,'dpd_90_ever':dpd_90_ever,
        'written_off_ever':written_off_ever,'settled_ever':settled_ever,
        'num_delinquent_now':num_delinquent_now,
        'home_loan_flag':home_loan_flag,'home_loan_emi':home_loan_emi,
        'personal_loan_flag':personal_loan_flag,'personal_loan_emi':personal_loan_emi,
        'auto_loan_flag':auto_loan_flag,'auto_loan_emi':auto_loan_emi,
        'total_existing_emi':total_existing_emi,'foir':foir,
        'num_credit_cards_existing':num_credit_cards_existing,
        'total_card_limit_existing':total_card_limit_existing,
        'banking_vintage_months':banking_vintage_months,
        'avg_monthly_balance':avg_monthly_balance,'min_monthly_balance':min_monthly_balance,
        'balance_stability':balance_stability,'fixed_deposit_flag':fixed_deposit_flag,
        'fd_amount':fd_amount,'investment_flag':investment_flag,
        'premium_banking_flag':premium_banking_flag,
        'internal_app_score':internal_app_score,'bureau_score_segment':bureau_score_segment,
        'debt_to_income_ratio':debt_to_income_ratio,'repayment_capacity':repayment_capacity,
        'leverage_ratio':leverage_ratio,'credit_to_income':credit_to_income,
        'payment_track_score':payment_track_score,'relationship_score':relationship_score,
        'fraud_risk_score':fraud_risk_score,'policy_override_flag':policy_override_flag,
        'approved':approved,'credit_limit_assigned':credit_limit_assigned,
    })

VAR_DICT = [
    # Application
    ("application_channel","str","Application Data","Channel through which application was submitted: Branch/DSA/Online/Pre-approved","DSA"),
    ("product_applied","str","Application Data","Card product applied for: Classic/Gold/Platinum/Signature","Gold"),
    ("requested_limit","int","Application Data","Credit limit requested by applicant ($)","100000"),
    ("city_tier","str","Application Data","City tier of applicant's address: Tier1/Tier2/Tier3","Tier1"),
    ("state","str","Application Data","State of applicant's registered address","MH"),
    ("residence_type","str","Application Data","Residence ownership: Owned/Rented/Family-owned/Company-provided","Owned"),
    ("residence_stability","int","Application Data","Months at current address — proxy for residential stability","36"),
    ("existing_bank_rel","int","Application Data","1 if applicant has an existing relationship with issuing bank","1"),
    ("num_existing_products","int","Application Data","Number of products held with issuing bank (0 if no relationship)","2"),
    ("preferred_lang","str","Application Data","Language preference: English/Hindi/Regional","Hindi"),
    # Employment
    ("employer_category","str","Employment & Income","Employer type: PSU/Large-Private/SME/Govt/Self-Employed/Gig","Large-Private"),
    ("employment_type","str","Employment & Income","Employment category: Salaried/Self-Employed/Business-Owner/Professional","Salaried"),
    ("years_in_current_job","float","Employment & Income","Years with current employer — job stability signal","3.5"),
    ("total_work_exp_years","float","Employment & Income","Total work experience in years across all jobs","7.5"),
    ("gross_annual_income","int","Employment & Income","Gross annual income as declared / verified ($)","720000"),
    ("net_monthly_income","int","Employment & Income","Estimated net monthly take-home income ($)","46800"),
    ("income_proof_type","str","Employment & Income","Document used for income verification: ITR/Form16/Bank-Statement/CA-Certificate/None","Form16"),
    ("income_verified_flag","int","Employment & Income","1 if income has been formally verified by underwriting","1"),
    ("salary_account_flag","int","Employment & Income","1 if salary is credited to an account with the issuing bank","1"),
    ("avg_monthly_credit","int","Employment & Income","Average monthly credits to bank account in last 6 months ($)","48000"),
    ("income_stability_score","float","Employment & Income","Score 0–100 measuring consistency of income inflow over 6 months","82.5"),
    ("employer_stability_score","float","Employment & Income","Score 0–100 reflecting employer's financial stability and reputation","74.0"),
    # Bureau
    ("bureau_score","int","Bureau Signals","Credit bureau score at time of application (300–850)","720"),
    ("bureau_score_vintage","int","Bureau Signals","Months since customer's first bureau record was created","24"),
    ("num_active_loans","int","Bureau Signals","Number of currently active loan accounts across all lenders","2"),
    ("num_active_cards","int","Bureau Signals","Number of currently active credit cards across all issuers","1"),
    ("num_closed_loans","int","Bureau Signals","Number of successfully closed loans in bureau history","4"),
    ("total_active_balance","int","Bureau Signals","Total outstanding balance across all active credit facilities ($)","350000"),
    ("total_credit_limit_bureau","int","Bureau Signals","Sum of all credit limits across all credit cards in bureau","200000"),
    ("bureau_utilization","float","Bureau Signals","Total bureau utilization: active_balance / total_card_limit (0–1.5)","0.42"),
    ("credit_age_oldest","int","Bureau Signals","Age in months of oldest credit account in bureau","72"),
    ("credit_age_newest","int","Bureau Signals","Age in months of most recently opened credit account","8"),
    ("credit_age_avg","int","Bureau Signals","Average age of all credit accounts in months","40"),
    ("num_inquiries_1m","int","Bureau Signals","Number of hard credit enquiries in last 1 month","1"),
    ("num_inquiries_3m","int","Bureau Signals","Number of hard credit enquiries in last 3 months","2"),
    ("num_inquiries_6m","int","Bureau Signals","Number of hard credit enquiries in last 6 months","3"),
    ("dpd_30_ever","int","Bureau Signals","1 if any 30+ DPD ever recorded across all bureau accounts","0"),
    ("dpd_60_ever","int","Bureau Signals","1 if any 60+ DPD ever recorded across all bureau accounts","0"),
    ("dpd_90_ever","int","Bureau Signals","1 if any 90+ DPD ever recorded across all bureau accounts","0"),
    ("written_off_ever","int","Bureau Signals","1 if any account was ever written off or charged off in bureau","0"),
    ("settled_ever","int","Bureau Signals","1 if any account was ever settled for less than full amount","0"),
    ("num_delinquent_now","int","Bureau Signals","Number of accounts currently delinquent in bureau","0"),
    # Obligations
    ("home_loan_flag","int","Existing Obligations","1 if applicant has an active home loan","1"),
    ("home_loan_emi","int","Existing Obligations","Monthly EMI for home loan ($, 0 if no home loan)","25000"),
    ("personal_loan_flag","int","Existing Obligations","1 if applicant has an active personal loan","0"),
    ("personal_loan_emi","int","Existing Obligations","Monthly EMI for personal loan ($, 0 if none)","0"),
    ("auto_loan_flag","int","Existing Obligations","1 if applicant has an active auto/vehicle loan","0"),
    ("auto_loan_emi","int","Existing Obligations","Monthly EMI for auto loan ($, 0 if none)","0"),
    ("total_existing_emi","int","Existing Obligations","Sum of all monthly EMI obligations across all loans ($)","25000"),
    ("foir","float","Existing Obligations","Fixed Obligation to Income Ratio: total_emi / net_monthly_income","0.53"),
    ("num_credit_cards_existing","int","Existing Obligations","Number of credit cards already held across all issuers","1"),
    ("total_card_limit_existing","int","Existing Obligations","Total credit limit across all existing cards ($)","150000"),
    # Banking Relationship
    ("banking_vintage_months","int","Banking Relationship","Months since account was opened with issuing bank (0 if no relationship)","36"),
    ("avg_monthly_balance","int","Banking Relationship","Average monthly balance in bank account last 6 months ($)","85000"),
    ("min_monthly_balance","int","Banking Relationship","Minimum monthly balance in bank account last 6 months ($)","42000"),
    ("balance_stability","float","Banking Relationship","Score 0–1 measuring stability of account balance over time","0.72"),
    ("fixed_deposit_flag","int","Banking Relationship","1 if applicant has an active fixed deposit with issuing bank","1"),
    ("fd_amount","int","Banking Relationship","Total fixed deposit amount with issuing bank ($, 0 if none)","200000"),
    ("investment_flag","int","Banking Relationship","1 if applicant has mutual funds/investments through issuing bank","0"),
    ("premium_banking_flag","int","Banking Relationship","1 if applicant is tagged as premium/priority banking customer","0"),
    # Derived
    ("internal_app_score","float","Risk Scores & Derived","Bank's internal application score (0–1000, higher = better)","680.5"),
    ("bureau_score_segment","str","Risk Scores & Derived","Bureau score band: Very Poor/Poor/Fair/Good/Excellent","Good"),
    ("debt_to_income_ratio","float","Risk Scores & Derived","Total bureau outstanding / gross annual income","0.49"),
    ("repayment_capacity","float","Risk Scores & Derived","(Net income - existing EMI) / Net income — free cash flow ratio (0–1)","0.47"),
    ("leverage_ratio","float","Risk Scores & Derived","Total active bureau balance / gross annual income","0.49"),
    ("credit_to_income","float","Risk Scores & Derived","Total bureau credit limit / gross annual income — leverage proxy","0.28"),
    ("payment_track_score","float","Risk Scores & Derived","Score 0–1 derived from DPD history — higher = cleaner track record","0.82"),
    ("relationship_score","float","Risk Scores & Derived","Score 0–1 measuring depth and quality of bank relationship","0.75"),
    ("fraud_risk_score","float","Risk Scores & Derived","Score 0–100 indicating probability of fraudulent application (lower = safer)","4.2"),
    ("policy_override_flag","int","Risk Scores & Derived","1 if underwriter manually overrode model recommendation","0"),
    # Targets
    ("approved","int","TARGET","1 if application was approved, 0 if declined","1"),
    ("credit_limit_assigned","int","TARGET","Credit limit assigned to approved applicants ($, 0 for declined)","125000"),
]

def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#14532d,#15803d);
                padding:32px;border-radius:12px;margin-bottom:24px'>
        <h1 style='color:white;margin:0;font-size:1.8rem'>
            💳 Capstone Project 2
        </h1>
        <h2 style='color:#86efac;margin:8px 0 0 0;font-size:1.2rem;font-weight:400'>
            Initial Credit Line Assignment: Approval & Limit Modelling
        </h2>
        <p style='color:#bbf7d0;margin:8px 0 0 0;font-size:0.85rem'>
            Independent Project &nbsp;|&nbsp; 30,000 Applicants &nbsp;|&nbsp;
            71 Variables &nbsp;|&nbsp; 2 Targets &nbsp;|&nbsp; ~36% Approval Rate
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Problem Statement",
        "📖 Variable Dictionary",
        "👁️ Data Preview",
        "⬇️ Download Dataset"
    ])

    with tab1:
        st.markdown("## 🏦 Business Context")
        st.markdown("""
When a customer applies for a credit card, two decisions happen in seconds:

**Decision 1 — Approve or Decline?**
The underwriting engine evaluates the applicant's creditworthiness. Too lenient → high defaults.
Too strict → good customers rejected, revenue lost. Getting this right is the core of credit underwriting.

**Decision 2 — How much credit to assign?**
For approved applicants, the bank must assign an initial credit limit. Too low → customer doesn't
use the card. Too high → over-leverages the customer, increases default risk.

Both decisions must balance **revenue** (approve more, assign higher limits) against
**risk** (minimize defaults and losses). This is the fundamental trade-off in credit underwriting.
        """)

        st.markdown("## 🎯 Problem Statement")
        st.success("""
You are a **Credit Underwriting Data Scientist** at a retail bank. The Head of Cards
Underwriting has tasked you with building a next-generation underwriting model:

*"Our current approval model uses a 15-variable scorecard built in 2020. Approval rates have
dropped to 34% but our DSA partners are complaining we're rejecting good customers. Meanwhile,
our NPAs on newly booked accounts are at 3.2%. Build me a two-part model: a better approval
decision model, and a credit limit assignment model that's correlated with long-term performance."*

**Your objective — Two models:**

**Part A — Approval Model (Classification)**
Predict which applicants should be approved (`approved = 1`).
Optimise for: maximize approvals while keeping expected default rate below 4%.

**Part B — Credit Limit Assignment Model (Regression)**
For approved applicants only, predict the appropriate credit limit (`credit_limit_assigned`).
Optimise for: assign limits that match customer's repayment capacity and income.
        """)

        st.markdown("## 📊 Dataset Description")
        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Applicants","30,000")
        col2.metric("Variables","71")
        col3.metric("Approval Rate","~36%")
        col4.metric("Avg Limit (Approved)","$144,889")
        col5.metric("Limit Range","$10K–$500K")

        st.markdown("""
**Applicant Universe:** Credit card applicants — new-to-bank customers and existing customers
applying for a new card. All data is sourced at time of application (no behavioral data).

**Variable Groups:**
| Group | Count | Description |
|---|---|---|
| Application Data | 10 | Channel, product, city tier, residence |
| Employment & Income | 12 | Employer type, income, verification, stability |
| Bureau Signals | 20 | Score, utilization, DPD history, inquiries |
| Existing Obligations | 10 | EMIs, FOIR, existing cards |
| Banking Relationship | 8 | Account vintage, balance, FD, premium flag |
| Risk Scores & Derived | 11 | App score, DTI, repayment capacity, fraud score |
| **Targets** | **2** | **approved, credit_limit_assigned** |
        """)

        st.markdown("## 🎯 Expected Deliverables")
        st.success("""
**1. Exploratory Data Analysis**
- Approval rate by key segments (bureau score band, FOIR bucket, city tier, employer type)
- Credit limit distribution for approved applicants
- Correlation between requested_limit vs credit_limit_assigned
- Missing value analysis and treatment

**2. Feature Engineering**
- Bucket continuous variables into risk bands (bureau score, FOIR, income)
- WoE encoding for approval model
- Interaction features (bureau_score × income_verified_flag, FOIR × employment_type)
- IV analysis for variable selection

**3. Part A — Approval Model (Classification)**
- Logistic Regression scorecard (primary — must be interpretable)
- XGBoost challenger model
- Evaluation: AUC-ROC, Gini, KS Statistic
- Approval rate vs expected default rate trade-off curve
- Optimal threshold: achieve >38% approval rate while keeping default rate <4%
- SHAP: top 10 drivers of rejection

**4. Part B — Credit Limit Model (Regression)**
- Subset to approved applicants only
- Linear Regression (interpretable baseline)
- XGBoost Regressor
- Evaluation: RMSE, MAE, R²
- Feature importance: what drives higher/lower limits?
- Sanity check: correlation between assigned limit and income

**5. Policy Rule Integration**
- Apply hard policy rules AFTER model score:
  - Auto-decline: written_off_ever = 1 OR dpd_90_ever = 1
  - Auto-cap limit: foir > 0.65 → limit ≤ 2× net monthly income
  - Policy override analysis: impact of policy_override_flag
- Show final approval rate after policy rules vs before

**6. Business Recommendation**
- Recommended score cutoff with approval rate and expected default rate
- Limit assignment formula (income multiplier approach + model adjustment)
- Revenue impact: expected card spend and interchange income per approval rate band
        """)

        st.markdown("## 📏 Evaluation Criteria")
        st.markdown("""
| Criterion | Weight |
|---|---|
| EDA — approval rate segmentation, limit distribution | 15% |
| WoE/IV analysis & variable selection | 15% |
| Part A: Approval model (Gini, KS, optimal threshold) | 25% |
| Part B: Limit model (RMSE, R², income correlation) | 20% |
| Policy rule integration & override analysis | 15% |
| Business recommendation (approval rate vs default trade-off) | 10% |
        """)

        st.markdown("## 📚 Key Concepts Reference")
        with st.expander("What is FOIR and why does it matter?"):
            st.markdown("""
**Fixed Obligation to Income Ratio (FOIR)** = Total monthly EMI / Net monthly income

- FOIR < 0.40: Low obligation — good repayment capacity
- FOIR 0.40–0.65: Moderate obligation — acceptable
- FOIR > 0.65: High obligation — risky, limit additional credit
- FOIR > 0.80: Typically auto-decline territory

Banks use FOIR as a hard policy rule AND as a model feature.
            """)
        with st.expander("What is the Approval Rate vs Default Rate Trade-off?"):
            st.markdown("""
At any score cutoff:
- **Lower cutoff** → Approve more → Higher revenue → Higher defaults
- **Higher cutoff** → Approve fewer → Lower defaults → Revenue lost

Your job is to find the **optimal cutoff** that meets the business target
(e.g. <4% expected default rate on new bookings) while maximising approvals.

Plot the trade-off curve: X-axis = approval rate, Y-axis = expected default rate.
The optimal point is where the default rate constraint is met.
            """)
        with st.expander("Income Multiplier for Credit Line Assignment"):
            st.markdown("""
Industry standard for initial credit line: **2–6× net monthly income**

- Conservative (new customer, no relationship): 2–3× NMI
- Standard: 3–4× NMI
- Premium (high bureau score, relationship banking): 4–6× NMI

Your model should output a limit that's consistent with this range.
Limits that are >6× NMI or <1× NMI should be flagged as outliers.
            """)

    with tab2:
        st.markdown("## 📖 Variable Dictionary")
        st.caption("71 variables + application_id + 2 targets. Search by name, group or description.")

        search = st.text_input("🔍 Search", placeholder="e.g. bureau, foir, income...")
        group_filter = st.selectbox("Filter by group",
                                    ["All Groups"] + sorted(set(v[2] for v in VAR_DICT)))
        filtered = VAR_DICT
        if search:
            filtered = [v for v in filtered
                        if search.lower() in v[0].lower() or search.lower() in v[3].lower()]
        if group_filter != "All Groups":
            filtered = [v for v in filtered if v[2] == group_filter]

        vdf = pd.DataFrame(filtered,
                           columns=["Variable Name","Type","Group","Description","Example Value"])

        def highlight(row):
            if row["Group"] == "TARGET":
                return ["background-color:#f0fdf4"]*len(row)
            return [""]*len(row)

        st.dataframe(vdf.style.apply(highlight, axis=1),
                     use_container_width=True, height=600)
        st.caption(f"Showing {len(filtered)} of {len(VAR_DICT)} variables")

    with tab3:
        st.markdown("## 👁️ Dataset Preview — First 50 Rows")
        with st.spinner("Generating dataset..."):
            df = generate_dataset()
        st.dataframe(df.head(50), use_container_width=True, height=420)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Applicants", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Approved", f"{df['approved'].sum():,} ({round(df['approved'].mean()*100,1)}%)")
        c4.metric("Avg Limit (Approved)",
                  f"${round(df[df['approved']==1]['credit_limit_assigned'].mean()):,}")
        c5.metric("Declined",
                  f"{(df['approved']==0).sum():,} ({round((df['approved']==0).mean()*100,1)}%)")

    with tab4:
        st.markdown("## ⬇️ Download Dataset")
        st.info("Enter your details to download. Access is logged for evaluation tracking.")
        st.divider()

        email  = st.text_input("Email", placeholder="you@company.com")
        emp_id = st.text_input("Employee ID", placeholder="EMP12345")

        if st.button("⬇️ Download CSV", type="primary"):
            if not email or "@" not in email:
                st.error("Enter a valid email address."); return
            if not emp_id.strip():
                st.error("Enter your Employee ID."); return
            with st.spinner("Generating dataset..."):
                df  = generate_dataset()
                csv = df.to_csv(index=False).encode("utf-8")
            log_download(email, emp_id)
            st.download_button(
                label="📥 Click here to download capstone2_creditline_dataset.csv",
                data=csv,
                file_name="capstone2_creditline_dataset.csv",
                mime="text/csv",
            )
            st.success(f"✅ Download ready! Logged for {email}")
            st.markdown(f"""
**Dataset details:**
- Rows: 30,000 applicants
- Columns: {len(df.columns)} (71 variables + application_id + approved + credit_limit_assigned)
- Approval rate: ~36%
- Avg credit limit (approved): ~$144,889
- Size: ~10 MB

**Targets:**
- `approved`: 1 = application approved, 0 = declined
- `credit_limit_assigned`: limit assigned in $ (0 for declined applicants — model on approved only)

**Important:** Build the limit model on approved applicants only (`approved == 1`).
            """)

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
