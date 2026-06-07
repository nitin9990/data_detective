import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

st.set_page_config(
    page_title="Capstone 3 — Long Term Default (Charge-off) Prediction",
    page_icon="⚠️", layout="wide"
)

def _sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def log_download(email, employee_id):
    try:
        _sb().table("attempts").insert({
            "email":       email.lower().strip(),
            "level":       "capstone3_chargeoff",
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
    np.random.seed(202); n = 25000

    mob                    = np.random.randint(3, 60, n)
    credit_vintage         = np.random.randint(6, 180, n)
    product_type           = np.random.choice(['Classic','Gold','Platinum','Signature'],n,p=[0.30,0.35,0.25,0.10])
    acquisition_channel    = np.random.choice(['Branch','DSA','Online','Referral'],n,p=[0.25,0.35,0.25,0.15])
    card_variant           = np.random.choice(['Standard','Rewards','Cashback','Travel'],n,p=[0.30,0.30,0.25,0.15])
    num_products_with_bank = np.random.randint(1,6,n)
    relationship_flag      = np.random.choice([0,1],n,p=[0.45,0.55])
    customer_segment       = np.random.choice(['Mass','Affluent','HNI'],n,p=[0.50,0.35,0.15])
    city_tier              = np.random.choice(['Tier1','Tier2','Tier3'],n,p=[0.35,0.35,0.30])
    state                  = np.random.choice(['MH','DL','KA','TN','UP','GJ','RJ','WB','AP','Others'],n)

    credit_limit           = np.random.randint(10000,500000,n)
    current_balance        = np.random.randint(0,500000,n)
    utilization_current    = np.round(np.clip(current_balance/credit_limit,0,1.2),3)
    min_pay_ratio_3m       = np.round(np.random.beta(3,2,n),3)
    payment_ratio_3m       = np.round(np.random.beta(3,3,n),3)
    payment_ratio_6m       = np.round(np.random.beta(3,3,n),3)
    revolving_balance      = np.random.randint(0,300000,n)
    cash_advance_ratio_3m  = np.round(np.random.beta(1,8,n),3)
    overlimit_count_6m     = np.random.randint(0,6,n)
    credit_limit_increase  = np.random.randint(0,3,n)
    balance_growth_6m      = np.round(np.random.normal(0.1,0.3,n),3)
    interest_paid_6m       = np.random.randint(0,50000,n)

    bureau_score           = np.random.randint(300,850,n)
    bureau_score_6m_ago    = bureau_score + np.random.randint(-80,30,n)
    bureau_score_change_6m = bureau_score - bureau_score_6m_ago
    num_active_loans       = np.random.randint(0,8,n)
    num_active_cards       = np.random.randint(1,6,n)
    total_bureau_balance   = np.random.randint(0,5000000,n)
    bureau_dpd_ever        = np.random.choice([0,1],n,p=[0.60,0.40])
    bureau_dpd_max_ever    = np.where(bureau_dpd_ever, np.random.randint(30,180,n), 0)
    num_delinquent_bureau  = np.random.randint(0,4,n)
    num_written_off_bureau = np.random.randint(0,2,n)
    credit_age_oldest      = np.random.randint(12,240,n)
    num_inquiries_6m       = np.random.randint(0,12,n)

    missed_payment_count_12m = np.random.randint(0,12,n)
    consecutive_missed       = np.random.randint(0,6,n)
    max_dpd_12m              = np.random.choice([0,30,60,90,120],n,p=[0.45,0.25,0.15,0.10,0.05])
    max_dpd_24m              = np.maximum(max_dpd_12m, np.random.choice([0,30,60],n,p=[0.50,0.30,0.20]))
    times_30dpd_ever         = np.random.randint(0,8,n)
    times_60dpd_ever         = np.clip(times_30dpd_ever-np.random.randint(0,3,n),0,None)
    times_90dpd_ever         = np.clip(times_60dpd_ever-np.random.randint(0,3,n),0,None)
    auto_debit_flag          = np.random.choice([0,1],n,p=[0.45,0.55])
    emi_bounce_count_6m      = np.random.randint(0,4,n)
    min_pay_streak           = np.random.randint(0,12,n)

    balance_3m_ago         = np.random.randint(0,400000,n)
    balance_6m_ago         = np.random.randint(0,400000,n)
    balance_trend          = np.round((current_balance-balance_6m_ago)/(balance_6m_ago+1),3)
    utilization_3m_ago     = np.round(np.clip(balance_3m_ago/credit_limit,0,1),3)
    utilization_trend_3m   = np.round(utilization_current-utilization_3m_ago,3)
    fees_charged_6m        = np.random.randint(0,15000,n)
    interest_rate_applied  = np.round(np.random.uniform(0.24,0.42,n),3)
    principal_at_risk      = current_balance.copy()
    expected_loss_amount   = (principal_at_risk*np.random.uniform(0.3,0.8,n)).astype(int)
    ltv_ratio              = np.round(current_balance/(credit_limit+1),3).clip(0,2)

    avg_txn_value_3m       = np.random.randint(500,20000,n)
    txn_count_3m           = np.random.randint(0,50,n)
    spend_drop_flag        = np.random.choice([0,1],n,p=[0.55,0.45])
    card_inactive_months   = np.random.randint(0,12,n)
    cash_advance_count_6m  = np.random.randint(0,10,n)
    declined_txn_ratio     = np.round(np.random.beta(1,6,n),3)
    merchant_cat_diversity = np.random.randint(1,15,n)
    intl_spend_flag        = np.random.choice([0,1],n,p=[0.80,0.20])

    income_estimate        = np.random.randint(150000,5000000,n)
    salary_credit_flag     = np.random.choice([0,1],n,p=[0.30,0.70])
    avg_salary_credit_6m   = salary_credit_flag*np.random.randint(15000,300000,n)
    salary_drop_flag       = np.random.choice([0,1],n,p=[0.65,0.35])
    employer_category      = np.random.choice(['PSU','Private','Self-Employed','Govt','Gig'],n,p=[0.15,0.45,0.20,0.10,0.10])
    income_stability_score = np.round(np.random.beta(4,2,n)*100,1)
    foir                   = np.round(np.random.uniform(0.1,0.8,n),3)
    loan_to_income_ratio   = np.round(np.random.uniform(0,6,n),3)

    gdp_growth_rate        = np.round(np.random.normal(6.5,1.5,n),2)
    repo_rate              = np.round(np.random.uniform(4.0,7.0,n),2)
    unemployment_rate      = np.round(np.random.uniform(5.0,9.0,n),2)
    inflation_rate         = np.round(np.random.uniform(3.0,8.0,n),2)
    credit_growth_rate     = np.round(np.random.normal(12.0,3.0,n),2)
    npa_industry_rate      = np.round(np.random.uniform(2.0,6.0,n),2)

    ews_score              = np.round(np.random.uniform(0,100,n),1)
    risk_flag_count        = np.random.randint(0,8,n)
    collection_contact_flag= np.random.choice([0,1],n,p=[0.70,0.30])
    restructure_flag       = np.random.choice([0,1],n,p=[0.88,0.12])
    hardship_flag          = np.random.choice([0,1],n,p=[0.85,0.15])
    balance_transfer_flag  = np.random.choice([0,1],n,p=[0.80,0.20])
    limit_decrease_flag    = np.random.choice([0,1],n,p=[0.85,0.15])
    adverse_media_flag     = np.random.choice([0,1],n,p=[0.96,0.04])

    internal_risk_score    = np.round(np.random.uniform(0,1000,n),1)
    pd_score_current       = np.round(np.random.beta(2,8,n),4)
    lgd_estimate           = np.round(np.random.uniform(0.4,0.9,n),3)
    ead_estimate           = current_balance.copy()
    expected_credit_loss   = np.round(pd_score_current*lgd_estimate*ead_estimate).astype(int)
    vintage_default_rate   = np.round(np.random.uniform(0.02,0.18,n),4)
    segment_default_rate   = np.round(np.random.uniform(0.03,0.15,n),4)
    model_score_v2         = np.round(np.random.uniform(0,100,n),1)

    log_odds = (
        -6.2
        + 2.0*(utilization_current.clip(0,1))
        + 1.8*(missed_payment_count_12m/12)
        + 2.5*(times_90dpd_ever/8)
        + 1.5*(num_written_off_bureau)
        + 1.2*(consecutive_missed/6)
        + 1.0*(balance_trend.clip(-1,2))
        + 1.5*(cash_advance_ratio_3m)
        - 1.5*(payment_ratio_3m)
        - 1.0*(bureau_score-300)/550
        - 0.8*(salary_credit_flag)
        - 0.6*(income_stability_score/100)
        + 0.5*(emi_bounce_count_6m/4)
        + 0.8*(foir.clip(0,1))
        + np.random.normal(0,0.8,n)
    )
    prob = 1/(1+np.exp(-log_odds))
    charge_off_24m = (np.random.uniform(0,1,n)<prob).astype(int)

    return pd.DataFrame({
        'customer_id':range(200001,225001),
        'mob':mob,'credit_vintage':credit_vintage,'product_type':product_type,
        'acquisition_channel':acquisition_channel,'card_variant':card_variant,
        'num_products_with_bank':num_products_with_bank,'relationship_flag':relationship_flag,
        'customer_segment':customer_segment,'city_tier':city_tier,'state':state,
        'credit_limit':credit_limit,'current_balance':current_balance,
        'utilization_current':utilization_current,'min_pay_ratio_3m':min_pay_ratio_3m,
        'payment_ratio_3m':payment_ratio_3m,'payment_ratio_6m':payment_ratio_6m,
        'revolving_balance':revolving_balance,'cash_advance_ratio_3m':cash_advance_ratio_3m,
        'overlimit_count_6m':overlimit_count_6m,'credit_limit_increase':credit_limit_increase,
        'balance_growth_6m':balance_growth_6m,'interest_paid_6m':interest_paid_6m,
        'bureau_score':bureau_score,'bureau_score_6m_ago':bureau_score_6m_ago,
        'bureau_score_change_6m':bureau_score_change_6m,'num_active_loans':num_active_loans,
        'num_active_cards':num_active_cards,'total_bureau_balance':total_bureau_balance,
        'bureau_dpd_ever':bureau_dpd_ever,'bureau_dpd_max_ever':bureau_dpd_max_ever,
        'num_delinquent_bureau':num_delinquent_bureau,'num_written_off_bureau':num_written_off_bureau,
        'credit_age_oldest':credit_age_oldest,'num_inquiries_6m':num_inquiries_6m,
        'missed_payment_count_12m':missed_payment_count_12m,'consecutive_missed':consecutive_missed,
        'max_dpd_12m':max_dpd_12m,'max_dpd_24m':max_dpd_24m,
        'times_30dpd_ever':times_30dpd_ever,'times_60dpd_ever':times_60dpd_ever,
        'times_90dpd_ever':times_90dpd_ever,'auto_debit_flag':auto_debit_flag,
        'emi_bounce_count_6m':emi_bounce_count_6m,'min_pay_streak':min_pay_streak,
        'balance_3m_ago':balance_3m_ago,'balance_6m_ago':balance_6m_ago,
        'balance_trend':balance_trend,'utilization_3m_ago':utilization_3m_ago,
        'utilization_trend_3m':utilization_trend_3m,'fees_charged_6m':fees_charged_6m,
        'interest_rate_applied':interest_rate_applied,'principal_at_risk':principal_at_risk,
        'expected_loss_amount':expected_loss_amount,'ltv_ratio':ltv_ratio,
        'avg_txn_value_3m':avg_txn_value_3m,'txn_count_3m':txn_count_3m,
        'spend_drop_flag':spend_drop_flag,'card_inactive_months':card_inactive_months,
        'cash_advance_count_6m':cash_advance_count_6m,'declined_txn_ratio':declined_txn_ratio,
        'merchant_cat_diversity':merchant_cat_diversity,'international_spend_flag':intl_spend_flag,
        'income_estimate':income_estimate,'salary_credit_flag':salary_credit_flag,
        'avg_salary_credit_6m':avg_salary_credit_6m,'salary_drop_flag':salary_drop_flag,
        'employer_category':employer_category,'income_stability_score':income_stability_score,
        'foir':foir,'loan_to_income_ratio':loan_to_income_ratio,
        'gdp_growth_rate':gdp_growth_rate,'repo_rate':repo_rate,
        'unemployment_rate':unemployment_rate,'inflation_rate':inflation_rate,
        'credit_growth_rate':credit_growth_rate,'npa_industry_rate':npa_industry_rate,
        'ews_score':ews_score,'risk_flag_count':risk_flag_count,
        'collection_contact_flag':collection_contact_flag,'restructure_flag':restructure_flag,
        'hardship_flag':hardship_flag,'balance_transfer_flag':balance_transfer_flag,
        'limit_decrease_flag':limit_decrease_flag,'adverse_media_flag':adverse_media_flag,
        'internal_risk_score':internal_risk_score,'pd_score_current':pd_score_current,
        'lgd_estimate':lgd_estimate,'ead_estimate':ead_estimate,
        'expected_credit_loss':expected_credit_loss,'vintage_default_rate':vintage_default_rate,
        'segment_default_rate':segment_default_rate,'model_score_v2':model_score_v2,
        'charge_off_24m':charge_off_24m,
    })

VAR_DICT = [
    # Account Vintage
    ("mob","int","Account Vintage & History","Months on book — how long customer has held this card","18"),
    ("credit_vintage","int","Account Vintage & History","Months since customer's oldest credit account was opened","72"),
    ("product_type","str","Account Vintage & History","Card product tier: Classic/Gold/Platinum/Signature","Gold"),
    ("acquisition_channel","str","Account Vintage & History","Customer acquisition channel: Branch/DSA/Online/Referral","DSA"),
    ("card_variant","str","Account Vintage & History","Card variant: Standard/Rewards/Cashback/Travel","Rewards"),
    ("num_products_with_bank","int","Account Vintage & History","Total products held with issuing bank","2"),
    ("relationship_flag","int","Account Vintage & History","1 if customer has savings/current account with same bank","1"),
    ("customer_segment","str","Account Vintage & History","Bank CRM segment: Mass/Affluent/HNI","Affluent"),
    ("city_tier","str","Account Vintage & History","City classification: Tier1/Tier2/Tier3","Tier1"),
    ("state","str","Account Vintage & History","State of customer's registered address","MH"),
    # Credit Behaviour
    ("credit_limit","int","Current Credit Behaviour","Sanctioned credit limit on the card ($)","150000"),
    ("current_balance","int","Current Credit Behaviour","Current outstanding balance on the card ($)","85000"),
    ("utilization_current","float","Current Credit Behaviour","Current credit utilization ratio (balance/limit), max capped at 1.2","0.57"),
    ("min_pay_ratio_3m","float","Current Credit Behaviour","Ratio of minimum payment made to minimum due in last 3 months","0.85"),
    ("payment_ratio_3m","float","Current Credit Behaviour","Ratio of total amount paid to total amount due in last 3 months","0.42"),
    ("payment_ratio_6m","float","Current Credit Behaviour","Ratio of total amount paid to total amount due in last 6 months","0.38"),
    ("revolving_balance","int","Current Credit Behaviour","Balance carried forward from previous cycles ($)","45000"),
    ("cash_advance_ratio_3m","float","Current Credit Behaviour","Cash advances as proportion of total credit usage in last 3M","0.08"),
    ("overlimit_count_6m","int","Current Credit Behaviour","Number of times balance exceeded credit limit in last 6 months","1"),
    ("credit_limit_increase","int","Current Credit Behaviour","Number of credit limit increases received in card lifetime","1"),
    ("balance_growth_6m","float","Current Credit Behaviour","Proportional growth in balance over last 6 months","0.15"),
    ("interest_paid_6m","int","Current Credit Behaviour","Total interest/finance charges paid in last 6 months ($)","12000"),
    # Bureau
    ("bureau_score","int","Bureau Signals","Current credit bureau score (300–850)","640"),
    ("bureau_score_6m_ago","int","Bureau Signals","Bureau score 6 months ago","670"),
    ("bureau_score_change_6m","int","Bureau Signals","Change in bureau score over last 6 months (negative = deterioration)","-30"),
    ("num_active_loans","int","Bureau Signals","Number of active loan facilities across all lenders","3"),
    ("num_active_cards","int","Bureau Signals","Number of active credit cards across all issuers","2"),
    ("total_bureau_balance","int","Bureau Signals","Total outstanding balance across all credit facilities ($)","450000"),
    ("bureau_dpd_ever","int","Bureau Signals","1 if any DPD ever recorded in bureau history","1"),
    ("bureau_dpd_max_ever","int","Bureau Signals","Maximum DPD ever observed across all bureau accounts","60"),
    ("num_delinquent_bureau","int","Bureau Signals","Number of currently delinquent accounts in bureau","1"),
    ("num_written_off_bureau","int","Bureau Signals","Number of accounts previously written off in bureau","0"),
    ("credit_age_oldest","int","Bureau Signals","Age in months of the oldest credit account in bureau","84"),
    ("num_inquiries_6m","int","Bureau Signals","Number of credit inquiries (hard pulls) in last 6 months","3"),
    # Payment History
    ("missed_payment_count_12m","int","Payment History (Longitudinal)","Number of missed payments in last 12 months","2"),
    ("consecutive_missed","int","Payment History (Longitudinal)","Current streak of consecutive missed payments","1"),
    ("max_dpd_12m","int","Payment History (Longitudinal)","Maximum DPD observed on this card in last 12 months","30"),
    ("max_dpd_24m","int","Payment History (Longitudinal)","Maximum DPD observed on this card in last 24 months","60"),
    ("times_30dpd_ever","int","Payment History (Longitudinal)","Lifetime count of 30+ DPD events on this card","3"),
    ("times_60dpd_ever","int","Payment History (Longitudinal)","Lifetime count of 60+ DPD events on this card","1"),
    ("times_90dpd_ever","int","Payment History (Longitudinal)","Lifetime count of 90+ DPD events on this card","0"),
    ("auto_debit_flag","int","Payment History (Longitudinal)","1 if auto-debit/ECS mandate is currently active","1"),
    ("emi_bounce_count_6m","int","Payment History (Longitudinal)","Number of EMI/ECS bounces in last 6 months","0"),
    ("min_pay_streak","int","Payment History (Longitudinal)","Consecutive months where only minimum payment was made","3"),
    # Balance Trends
    ("balance_3m_ago","int","Balance & Exposure Trends","Card balance 3 months ago ($)","70000"),
    ("balance_6m_ago","int","Balance & Exposure Trends","Card balance 6 months ago ($)","55000"),
    ("balance_trend","float","Balance & Exposure Trends","Proportional change in balance vs 6 months ago (positive = growing)","0.55"),
    ("utilization_3m_ago","float","Balance & Exposure Trends","Credit utilization ratio 3 months ago","0.47"),
    ("utilization_trend_3m","float","Balance & Exposure Trends","Change in utilization in last 3 months (positive = increasing)","0.10"),
    ("fees_charged_6m","int","Balance & Exposure Trends","Total late fees and penalty charges in last 6 months ($)","2400"),
    ("interest_rate_applied","float","Balance & Exposure Trends","Effective annual interest rate currently applied on the card","0.36"),
    ("principal_at_risk","int","Balance & Exposure Trends","Principal balance at risk of being written off ($)","85000"),
    ("expected_loss_amount","int","Balance & Exposure Trends","Estimated loss amount given default ($)","51000"),
    ("ltv_ratio","float","Balance & Exposure Trends","Loan-to-value ratio: balance/credit limit (can exceed 1 if overlimit)","0.57"),
    # Transactions
    ("avg_txn_value_3m","int","Transaction Patterns","Average value per card transaction in last 3 months ($)","3500"),
    ("txn_count_3m","int","Transaction Patterns","Total number of card transactions in last 3 months","12"),
    ("spend_drop_flag","int","Transaction Patterns","1 if spending in last 3M dropped >30% vs prior 3M","0"),
    ("card_inactive_months","int","Transaction Patterns","Number of months card had zero transactions in last 12M","2"),
    ("cash_advance_count_6m","int","Transaction Patterns","Number of cash advance transactions in last 6 months","3"),
    ("declined_txn_ratio","float","Transaction Patterns","Proportion of transactions declined due to insufficient limit/funds","0.05"),
    ("merchant_cat_diversity","int","Transaction Patterns","Number of distinct merchant categories transacted in last 3M","5"),
    ("international_spend_flag","int","Transaction Patterns","1 if international transactions made in last 6 months","0"),
    # Income
    ("income_estimate","int","Income & Employment","Estimated annual income based on salary credits and bureau ($)","720000"),
    ("salary_credit_flag","int","Income & Employment","1 if regular salary credits visible in bank account","1"),
    ("avg_salary_credit_6m","int","Income & Employment","Average salary credit amount in last 6 months ($, 0 if no salary)","60000"),
    ("salary_drop_flag","int","Income & Employment","1 if salary credit amount dropped >20% in last 3 months","0"),
    ("employer_category","str","Income & Employment","Employer type: PSU/Private/Self-Employed/Govt/Gig","Private"),
    ("income_stability_score","float","Income & Employment","Score 0–100 measuring consistency of income inflows","78.5"),
    ("foir","float","Income & Employment","Fixed Obligation to Income Ratio — total EMI / monthly income","0.45"),
    ("loan_to_income_ratio","float","Income & Employment","Total outstanding loans / annual income","1.8"),
    # Macro
    ("gdp_growth_rate","float","Macroeconomic Indicators","GDP growth rate (%) at time of observation","6.8"),
    ("repo_rate","float","Macroeconomic Indicators","RBI repo rate (%) at time of observation — affects credit cost","6.5"),
    ("unemployment_rate","float","Macroeconomic Indicators","National unemployment rate (%) at time of observation","6.2"),
    ("inflation_rate","float","Macroeconomic Indicators","CPI inflation rate (%) at time of observation","5.1"),
    ("credit_growth_rate","float","Macroeconomic Indicators","Industry-level credit growth rate (%) — proxy for economic expansion","13.2"),
    ("npa_industry_rate","float","Macroeconomic Indicators","Industry NPA rate (%) — systemic credit stress indicator","3.8"),
    # Early Warning
    ("ews_score","float","Early Warning Signals","Internal early warning score (0–100, higher = more risk)","45.0"),
    ("risk_flag_count","int","Early Warning Signals","Number of active risk triggers on the account","2"),
    ("collection_contact_flag","int","Early Warning Signals","1 if customer received a collections contact in last 6M","0"),
    ("restructure_flag","int","Early Warning Signals","1 if loan restructuring was done in last 24 months","0"),
    ("hardship_flag","int","Early Warning Signals","1 if customer has declared financial hardship","0"),
    ("balance_transfer_flag","int","Early Warning Signals","1 if balance was transferred from another card in last 12M","0"),
    ("limit_decrease_flag","int","Early Warning Signals","1 if credit limit was proactively reduced by bank","0"),
    ("adverse_media_flag","int","Early Warning Signals","1 if adverse news/media coverage detected for customer","0"),
    # Derived
    ("internal_risk_score","float","Derived / Risk Scores","Bank's internal risk score (0–1000, higher = riskier)","420.0"),
    ("pd_score_current","float","Derived / Risk Scores","Current probability of default estimate from existing model (0–1)","0.08"),
    ("lgd_estimate","float","Derived / Risk Scores","Loss Given Default estimate — fraction of balance expected to be lost","0.65"),
    ("ead_estimate","int","Derived / Risk Scores","Exposure at Default — balance expected at time of default ($)","85000"),
    ("expected_credit_loss","int","Derived / Risk Scores","ECL = PD × LGD × EAD — expected loss in dollars ($)","4420"),
    ("vintage_default_rate","float","Derived / Risk Scores","Default rate for this customer's acquisition vintage","0.08"),
    ("segment_default_rate","float","Derived / Risk Scores","Historical default rate for this customer's risk segment","0.07"),
    ("model_score_v2","float","Derived / Risk Scores","Previous model score (v2) for benchmarking (0–100, higher = riskier)","38.5"),
    # Target
    ("charge_off_24m","int","TARGET","1 if customer charged off (180+ DPD, account written off) within 24 months","0"),
]

def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#7f1d1d,#b91c1c);
                padding:32px;border-radius:12px;margin-bottom:24px'>
        <h1 style='color:white;margin:0;font-size:1.8rem'>
            ⚠️ Capstone Project 3
        </h1>
        <h2 style='color:#fca5a5;margin:8px 0 0 0;font-size:1.2rem;font-weight:400'>
            Long-Term Default Prediction: Charge-off within 24 Months
        </h2>
        <p style='color:#fecaca;margin:8px 0 0 0;font-size:0.85rem'>
            Independent Project &nbsp;|&nbsp; 25,000 Customers &nbsp;|&nbsp;
            92 Variables &nbsp;|&nbsp; 1 Target &nbsp;|&nbsp; ~10% Charge-off Rate
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
A charge-off occurs when a bank declares a credit card debt unlikely to be collected and writes
it off as a loss — typically after 180+ days past due. For a bank with a $2B credit card
portfolio, a 1% reduction in charge-off rate saves $20M annually.

The challenge: charge-offs happen 12–24 months after the early warning signals appear. By the
time a customer is 90 DPD, it's often too late for preventive action. Banks need to identify
**high-risk customers 18–24 months in advance** — while the account is still current or
mildly delinquent — and intervene with:

- Proactive credit limit management
- Early restructuring offers
- Enhanced monitoring and early collections
- Provisioning and capital reserve planning (IFRS 9 / ECL requirements)
        """)

        st.markdown("## 🎯 Problem Statement")
        st.error("""
You are a **Senior Credit Risk Modeller** at a retail bank. The Chief Risk Officer (CRO) has
mandated a next-generation charge-off prediction model to replace the bank's legacy scorecard.

The mandate:

*"Our current model was built in 2019 and has a Gini of 0.52. It misses 35% of charge-offs
in the 12-month window. With IFRS 9 requirements, we need a model that identifies high-risk
customers early enough to set Stage 2 and Stage 3 provisions. Build a 24-month charge-off
prediction model with Gini > 0.65, and provide SHAP-based explainability for regulatory audit."*

**Your objective:**
Predict which active credit card customers will **charge off within the next 24 months**
(`charge_off_24m = 1`) using a combination of credit behaviour, bureau signals, macroeconomic
indicators and early warning signals.
        """)

        st.markdown("## 📊 Dataset Description")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Customers", "25,000")
        col2.metric("Variables", "92")
        col3.metric("Charge-off Rate", "~10%")
        col4.metric("Forecast Horizon", "24 Months")
        col5.metric("Class Imbalance", "1:9")

        st.markdown("""
**Customer Universe:** Active credit card customers (current and 1–30 DPD) observed at a
single point in time. Target is whether they charged off in the subsequent 24 months.

**Variable Groups:**
| Group | Count | Description |
|---|---|---|
| Account Vintage & History | 10 | MOB, product type, relationship, segment |
| Current Credit Behaviour | 12 | Balance, utilization, payment ratios |
| Bureau Signals | 12 | Score, multi-lender exposure, DPD history |
| Payment History (Longitudinal) | 10 | DPD events, missed payments, streaks |
| Balance & Exposure Trends | 10 | 3M/6M balance trends, fees, ECL components |
| Transaction Patterns | 8 | Spend, cash advances, card activity |
| Income & Employment | 8 | Salary credits, FOIR, employer category |
| Macroeconomic Indicators | 6 | GDP, repo rate, unemployment, NPA rate |
| Early Warning Signals | 8 | EWS score, risk flags, restructure, hardship |
| Derived / Risk Scores | 8 | PD, LGD, EAD, ECL, vintage default rate |
| **Target** | **1** | **charge_off_24m** |
        """)

        st.markdown("## 🎯 Expected Deliverables")
        st.success("""
**1. Exploratory Data Analysis**
- Target distribution and class imbalance analysis
- Charge-off rate by key segments (product type, MOB bucket, bureau score band, city tier)
- Variable distributions — identify key risk signals vs noise
- Correlation heatmap — identify multicollinearity
- Vintage curve analysis: charge-off rate by months-on-book bucket

**2. Data Preprocessing**
- Missing value treatment strategy with justification
- Outlier treatment (IQR capping / Winsorization)
- Categorical encoding (WoE encoding preferred for credit scorecards)
- Variable reduction: IV (Information Value) filtering — drop variables with IV < 0.02

**3. Feature Engineering**
- Create interaction features (e.g. utilization × bureau_score_change_6m)
- Bucket continuous variables into risk bands where applicable
- Compute WoE and IV for all variables — shortlist top 30

**4. Model Development**
- Logistic Regression (scorecard-style — interpretable baseline)
- XGBoost / LightGBM (champion model)
- Handle class imbalance: SMOTE or class_weight
- Evaluation: AUC-ROC, Gini, KS Statistic, PSI (Population Stability Index)

**5. SHAP Explainability**
- Global: Top 15 features by mean |SHAP|
- Local: Waterfall chart for 3 sample customers (1 charge-off, 1 near-miss, 1 low-risk)
- Directional consistency check: do SHAP directions match domain knowledge?

**6. IFRS 9 / Staging Application**
- Apply model scores to classify accounts into:
  - Stage 1: Low risk (PD < 5%) — 12-month ECL provision
  - Stage 2: Significant increase in credit risk (PD 5–15%) — Lifetime ECL
  - Stage 3: Credit-impaired (PD > 15%) — Lifetime ECL, full provision
- Calculate total ECL provision requirement for the 25,000-customer portfolio

**7. Model Validation & Champion-Challenger**
- Out-of-time validation (last 6 months as holdout)
- Comparison vs existing model_score_v2 benchmark
- Stability report: PSI across time buckets
        """)

        st.markdown("## 📏 Evaluation Criteria")
        st.markdown("""
| Criterion | Weight |
|---|---|
| EDA quality — vintage curves, segment analysis | 15% |
| IV/WoE analysis & variable selection | 15% |
| Model performance — Gini > 0.65 target | 25% |
| SHAP explainability & directional validation | 15% |
| IFRS 9 staging & ECL calculation | 15% |
| Champion-Challenger comparison | 10% |
| Code quality, documentation, reproducibility | 5% |
        """)

        st.markdown("## 📚 Key Concepts Reference")
        with st.expander("What is Gini Coefficient?"):
            st.markdown("""
**Gini = 2 × AUC − 1**
- Target: Gini > 0.65 (AUC > 0.825)
- Existing model: Gini = 0.52 (AUC = 0.76) — your benchmark to beat
- Gini > 0.5 = acceptable; > 0.65 = good; > 0.75 = excellent for credit risk
            """)
        with st.expander("What is KS Statistic?"):
            st.markdown("""
**KS = max separation between cumulative distributions of charge-offs vs non-charge-offs**
- KS > 40% = good discriminatory power
- Used alongside Gini in credit risk model validation
- Report KS at decile 3–4 (where it typically peaks)
            """)
        with st.expander("What is Information Value (IV)?"):
            st.markdown("""
**IV measures the predictive power of each variable:**
- IV < 0.02: Not useful — drop
- 0.02–0.10: Weak predictor
- 0.10–0.30: Medium predictor
- > 0.30: Strong predictor
- > 0.50: Suspiciously strong — check for data leakage

WoE (Weight of Evidence) encoding + IV filtering is the industry standard for credit scorecard development.
            """)
        with st.expander("What is IFRS 9 Staging?"):
            st.markdown("""
**IFRS 9 requires banks to classify loans into 3 stages:**
- **Stage 1:** No significant increase in credit risk → 12-month ECL provision
- **Stage 2:** Significant increase in credit risk (SICR) → Lifetime ECL provision
- **Stage 3:** Credit-impaired (default) → Lifetime ECL, full provision

**ECL = PD × LGD × EAD**
- PD: Probability of Default (from your model)
- LGD: Loss Given Default (provided in dataset)
- EAD: Exposure at Default (current balance)

Your model's PD output directly feeds the bank's provisioning engine.
            """)
        with st.expander("What is PSI (Population Stability Index)?"):
            st.markdown("""
**PSI measures how much the score distribution has shifted between development and deployment:**
- PSI < 0.10: No significant shift — model is stable
- 0.10–0.25: Moderate shift — investigate
- PSI > 0.25: Major shift — model needs recalibration

Always report PSI in model validation — regulators require it.
            """)

    with tab2:
        st.markdown("## 📖 Variable Dictionary")
        st.caption("92 variables + customer_id + target. Search or filter by group.")

        search = st.text_input("🔍 Search variables",
                               placeholder="e.g. utilization, bureau, ews...")
        group_filter = st.selectbox("Filter by group",
                                    ["All Groups"] + sorted(set(v[2] for v in VAR_DICT)))

        filtered = VAR_DICT
        if search:
            filtered = [v for v in filtered if search.lower() in v[0].lower()
                        or search.lower() in v[3].lower()]
        if group_filter != "All Groups":
            filtered = [v for v in filtered if v[2] == group_filter]

        vdf = pd.DataFrame(filtered,
                           columns=["Variable Name","Type","Group","Description","Example Value"])

        def highlight(row):
            if row["Group"] == "TARGET":
                return ["background-color:#fef2f2"]*len(row)
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
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Charge-off Rate", f"{round(df['charge_off_24m'].mean()*100,1)}%")
        c4.metric("Charge-offs", f"{df['charge_off_24m'].sum():,}")
        c5.metric("Non Charge-offs", f"{(df['charge_off_24m']==0).sum():,}")

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
                label="📥 Click here to download capstone3_chargeoff_dataset.csv",
                data=csv,
                file_name="capstone3_chargeoff_dataset.csv",
                mime="text/csv",
            )
            st.success(f"✅ Download ready! Logged for {email}")
            st.markdown(f"""
**Dataset details:**
- Rows: 25,000 customers
- Columns: {len(df.columns)} (92 variables + customer_id + charge_off_24m)
- Charge-off rate: ~10% (realistic industry rate)
- Size: ~12 MB

**Target:**
- `charge_off_24m`: 1 if customer charged off (180+ DPD, written off) within 24 months

**Note:** Use `model_score_v2` as your benchmark to compare against. Beat it.
            """)

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
