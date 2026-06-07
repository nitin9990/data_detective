import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client

st.set_page_config(
    page_title="Capstone 1 — Collections: Willingness & Ability to Pay",
    page_icon="💼", layout="wide"
)

def _sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def log_download(email, employee_id):
    try:
        _sb().table("attempts").insert({
            "email":       email.lower().strip(),
            "level":       "capstone1_collections",
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
    np.random.seed(101); n=20000
    current_dpd=np.random.choice([30,60,90,120,150,180],n,p=[0.35,0.25,0.18,0.12,0.06,0.04])
    max_dpd_6m=current_dpd+np.random.randint(0,30,n)
    times_30dpd=np.random.randint(1,8,n)
    times_60dpd=np.clip(times_30dpd-np.random.randint(0,3,n),0,None)
    times_90dpd=np.clip(times_60dpd-np.random.randint(0,3,n),0,None)
    dpd_at_last_statement=np.random.choice([0,30,60,90],n,p=[0.1,0.5,0.25,0.15])
    mob=np.random.randint(3,72,n)
    days_since_first_delinq=np.random.randint(30,540,n)
    payment_ratio_3m=np.round(np.random.beta(2,5,n),3)
    payment_ratio_6m=np.round(np.random.beta(2,4,n),3)
    ptp_kept_rate=np.round(np.random.beta(2,3,n),3)
    ptp_broken_rate=np.round((1-ptp_kept_rate+np.random.normal(0,0.05,n)).clip(0,1),3)
    partial_payment_flag=np.random.choice([0,1],n,p=[0.55,0.45])
    last_payment_amount=np.random.randint(0,50000,n)
    months_since_last_payment=np.random.randint(1,12,n)
    payment_frequency=np.random.choice(['Never','Irregular','Monthly','Bi-monthly'],n,p=[0.2,0.35,0.3,0.15])
    min_pay_count=np.random.randint(0,6,n)
    full_pay_count=np.random.randint(0,3,n)
    auto_debit_flag=np.random.choice([0,1],n,p=[0.6,0.4])
    payment_channel=np.random.choice(['Online','Branch','UPI','NEFT','Cheque'],n,p=[0.4,0.1,0.3,0.15,0.05])
    total_outstanding=np.random.randint(5000,500000,n)
    principal_outstanding=(total_outstanding*np.random.uniform(0.6,0.8,n)).astype(int)
    interest_outstanding=(total_outstanding*np.random.uniform(0.1,0.25,n)).astype(int)
    fees_outstanding=total_outstanding-principal_outstanding-interest_outstanding
    credit_limit=np.random.randint(10000,500000,n)
    utilization=np.round(np.clip(total_outstanding/credit_limit,0,1),3)
    overlimit_flag=(utilization>1).astype(int)
    revolving_balance=np.random.randint(0,200000,n)
    settlement_offer_pct=np.round(np.random.uniform(0.4,0.9,n),2)
    write_off_risk_score=np.round(np.random.uniform(0,100,n),1)
    bureau_score=np.random.randint(300,800,n)
    bureau_score_change_3m=np.random.randint(-100,50,n)
    num_active_loans=np.random.randint(0,8,n)
    num_delinquent_accounts=np.random.randint(0,5,n)
    total_bureau_outstanding=np.random.randint(0,3000000,n)
    num_inquiries_3m=np.random.randint(0,10,n)
    secured_loan_flag=np.random.choice([0,1],n,p=[0.5,0.5])
    personal_loan_flag=np.random.choice([0,1],n,p=[0.6,0.4])
    home_loan_flag=np.random.choice([0,1],n,p=[0.7,0.3])
    loan_to_income_ratio=np.round(np.random.uniform(0,5,n),2)
    bureau_dpd_max=np.random.randint(0,180,n)
    credit_vintage=np.random.randint(6,120,n)
    num_calls_made=np.random.randint(0,20,n)
    num_calls_answered=np.clip(num_calls_made-np.random.randint(0,10,n),0,None)
    rpc_flag=(num_calls_answered>0).astype(int)
    ptp_flag=np.random.choice([0,1],n,p=[0.45,0.55])
    ptp_amount=(ptp_flag*np.random.randint(1000,50000,n))
    broken_ptp_count=np.random.randint(0,4,n)
    field_visit_flag=np.random.choice([0,1],n,p=[0.75,0.25])
    legal_notice_flag=np.random.choice([0,1],n,p=[0.80,0.20])
    settlement_offered_flag=np.random.choice([0,1],n,p=[0.6,0.4])
    last_contact_channel=np.random.choice(['Call','SMS','Email','WhatsApp','Field'],n,p=[0.4,0.25,0.1,0.2,0.05])
    days_since_last_contact=np.random.randint(0,30,n)
    contact_attempt_freq=np.random.randint(0,5,n)
    escalation_flag=np.random.choice([0,1],n,p=[0.7,0.3])
    agency_assigned_flag=np.random.choice([0,1],n,p=[0.65,0.35])
    collection_stage=np.random.choice(['Early','Mid','Late','Pre-Writeoff'],n,p=[0.35,0.3,0.25,0.1])
    avg_spend_3m=np.random.randint(0,80000,n).astype(float)
    spend_drop_flag=np.random.choice([0,1],n,p=[0.45,0.55])
    cash_advance_count_3m=np.random.randint(0,8,n)
    declined_txn_count=np.random.randint(0,10,n)
    atm_withdrawal_3m=np.random.randint(0,30000,n)
    online_spend_ratio=np.round(np.random.beta(3,2,n),3)
    merchant_cat_diversity=np.random.randint(1,15,n)
    spend_6m_vs_12m=np.round(np.random.normal(0.8,0.2,n),3)
    last_txn_days_ago=np.random.randint(0,90,n)
    card_active_flag=np.random.choice([0,1],n,p=[0.3,0.7])
    salary_credit_flag=np.random.choice([0,1],n,p=[0.35,0.65])
    avg_salary_credit_3m=(salary_credit_flag*np.random.randint(10000,200000,n))
    salary_credit_drop_flag=np.random.choice([0,1],n,p=[0.6,0.4])
    employer_category=np.random.choice(['PSU','Private','Self-Employed','Govt','Gig'],n,p=[0.15,0.45,0.2,0.1,0.1])
    salary_credit_months_cnt=np.random.randint(0,12,n)
    income_estimate=np.random.randint(100000,3000000,n)
    income_stability_score=np.round(np.random.beta(4,2,n)*100,1)
    gig_worker_flag=np.random.choice([0,1],n,p=[0.85,0.15])
    months_on_book=mob.copy()
    product_type=np.random.choice(['Classic','Gold','Platinum','Signature'],n,p=[0.3,0.35,0.25,0.1])
    card_variant=np.random.choice(['Standard','Rewards','Cashback','Travel'],n,p=[0.3,0.3,0.25,0.15])
    acquisition_channel=np.random.choice(['Branch','DSA','Online','Referral'],n,p=[0.25,0.35,0.25,0.15])
    relationship_banking_flag=np.random.choice([0,1],n,p=[0.55,0.45])
    num_products_with_bank=np.random.randint(1,6,n)
    cross_sell_flag=np.random.choice([0,1],n,p=[0.5,0.5])
    premium_customer_flag=np.random.choice([0,1],n,p=[0.7,0.3])
    vintage_segment=np.random.choice(['New','Growing','Mature','Long-term'],n,p=[0.2,0.3,0.3,0.2])
    customer_segment=np.random.choice(['Mass','Affluent','HNI'],n,p=[0.5,0.35,0.15])
    city_tier=np.random.choice(['Tier1','Tier2','Tier3'],n,p=[0.35,0.35,0.30])
    state=np.random.choice(['MH','DL','KA','TN','UP','GJ','RJ','WB','AP','Others'],n)
    contact_score=np.round(np.random.uniform(0,100,n),1)
    email_valid_flag=np.random.choice([0,1],n,p=[0.2,0.8])
    mobile_valid_flag=np.random.choice([0,1],n,p=[0.05,0.95])
    whatsapp_opt_in=np.random.choice([0,1],n,p=[0.35,0.65])
    preferred_contact_time=np.random.choice(['Morning','Afternoon','Evening'],n,p=[0.2,0.3,0.5])
    ability_score_raw=np.round(0.3*(income_estimate/income_estimate.max())+0.2*salary_credit_flag+0.2*(1-utilization.clip(0,1))+0.15*(income_stability_score/100)+0.15*(1-loan_to_income_ratio.clip(0,5)/5),3)
    willingness_score_raw=np.round((0.25*ptp_kept_rate+0.20*payment_ratio_3m+0.15*rpc_flag+0.15*(1-broken_ptp_count/4)+0.15*(auto_debit_flag*0.5+contact_score/200)+0.10*(1-months_since_last_payment/12)).clip(0,1),3)
    roll_forward_risk=np.round(np.random.beta(2,3,n),3)
    cure_probability_raw=np.round((ability_score_raw*0.5+willingness_score_raw*0.5)*np.random.uniform(0.8,1.2,n),3).clip(0,1)
    days_to_chargeoff_risk=np.random.randint(30,180,n)
    net_collectible_amount=(total_outstanding*settlement_offer_pct).astype(int)
    expected_recovery=(net_collectible_amount*cure_probability_raw).astype(int)
    will_pay_lo=(-1.0+2.0*willingness_score_raw+1.5*ptp_flag+1.2*ptp_kept_rate-1.0*ptp_broken_rate+0.8*payment_ratio_3m-0.5*(current_dpd/180)+0.5*rpc_flag+np.random.normal(0,0.5,n))
    will_pay=(np.random.uniform(0,1,n)<1/(1+np.exp(-will_pay_lo))).astype(int)
    can_pay_lo=(-0.5+2.5*ability_score_raw+1.0*salary_credit_flag-1.0*(current_dpd/180)+0.8*(income_stability_score/100)-0.5*loan_to_income_ratio.clip(0,5)/5+np.random.normal(0,0.5,n))
    can_pay=(np.random.uniform(0,1,n)<1/(1+np.exp(-can_pay_lo))).astype(int)

    return pd.DataFrame({
        'customer_id':range(100001,120001),
        'current_dpd':current_dpd,'max_dpd_6m':max_dpd_6m,
        'times_30dpd':times_30dpd,'times_60dpd':times_60dpd,'times_90dpd':times_90dpd,
        'dpd_at_last_statement':dpd_at_last_statement,'mob':mob,
        'days_since_first_delinq':days_since_first_delinq,
        'payment_ratio_3m':payment_ratio_3m,'payment_ratio_6m':payment_ratio_6m,
        'ptp_kept_rate':ptp_kept_rate,'ptp_broken_rate':ptp_broken_rate,
        'partial_payment_flag':partial_payment_flag,'last_payment_amount':last_payment_amount,
        'months_since_last_payment':months_since_last_payment,
        'payment_frequency':payment_frequency,'min_pay_count':min_pay_count,
        'full_pay_count':full_pay_count,'auto_debit_flag':auto_debit_flag,
        'payment_channel':payment_channel,
        'total_outstanding':total_outstanding,'principal_outstanding':principal_outstanding,
        'interest_outstanding':interest_outstanding,'fees_outstanding':fees_outstanding,
        'credit_limit':credit_limit,'utilization':utilization,'overlimit_flag':overlimit_flag,
        'revolving_balance':revolving_balance,'settlement_offer_pct':settlement_offer_pct,
        'write_off_risk_score':write_off_risk_score,
        'bureau_score':bureau_score,'bureau_score_change_3m':bureau_score_change_3m,
        'num_active_loans':num_active_loans,'num_delinquent_accounts':num_delinquent_accounts,
        'total_bureau_outstanding':total_bureau_outstanding,'num_inquiries_3m':num_inquiries_3m,
        'secured_loan_flag':secured_loan_flag,'personal_loan_flag':personal_loan_flag,
        'home_loan_flag':home_loan_flag,'loan_to_income_ratio':loan_to_income_ratio,
        'bureau_dpd_max':bureau_dpd_max,'credit_vintage':credit_vintage,
        'num_calls_made':num_calls_made,'num_calls_answered':num_calls_answered,
        'rpc_flag':rpc_flag,'ptp_flag':ptp_flag,'ptp_amount':ptp_amount,
        'broken_ptp_count':broken_ptp_count,'field_visit_flag':field_visit_flag,
        'legal_notice_flag':legal_notice_flag,'settlement_offered_flag':settlement_offered_flag,
        'last_contact_channel':last_contact_channel,
        'days_since_last_contact':days_since_last_contact,
        'contact_attempt_freq':contact_attempt_freq,'escalation_flag':escalation_flag,
        'agency_assigned_flag':agency_assigned_flag,'collection_stage':collection_stage,
        'avg_spend_3m':avg_spend_3m,'spend_drop_flag':spend_drop_flag,
        'cash_advance_count_3m':cash_advance_count_3m,'declined_txn_count':declined_txn_count,
        'atm_withdrawal_3m':atm_withdrawal_3m,'online_spend_ratio':online_spend_ratio,
        'merchant_cat_diversity':merchant_cat_diversity,'spend_6m_vs_12m':spend_6m_vs_12m,
        'last_txn_days_ago':last_txn_days_ago,'card_active_flag':card_active_flag,
        'salary_credit_flag':salary_credit_flag,'avg_salary_credit_3m':avg_salary_credit_3m,
        'salary_credit_drop_flag':salary_credit_drop_flag,'employer_category':employer_category,
        'salary_credit_months_cnt':salary_credit_months_cnt,'income_estimate':income_estimate,
        'income_stability_score':income_stability_score,'gig_worker_flag':gig_worker_flag,
        'months_on_book':months_on_book,'product_type':product_type,'card_variant':card_variant,
        'acquisition_channel':acquisition_channel,
        'relationship_banking_flag':relationship_banking_flag,
        'num_products_with_bank':num_products_with_bank,'cross_sell_flag':cross_sell_flag,
        'premium_customer_flag':premium_customer_flag,'vintage_segment':vintage_segment,
        'customer_segment':customer_segment,
        'city_tier':city_tier,'state':state,'contact_score':contact_score,
        'email_valid_flag':email_valid_flag,'mobile_valid_flag':mobile_valid_flag,
        'whatsapp_opt_in':whatsapp_opt_in,'preferred_contact_time':preferred_contact_time,
        'ability_score_raw':ability_score_raw,'willingness_score_raw':willingness_score_raw,
        'roll_forward_risk':roll_forward_risk,'cure_probability_raw':cure_probability_raw,
        'days_to_chargeoff_risk':days_to_chargeoff_risk,
        'net_collectible_amount':net_collectible_amount,'expected_recovery':expected_recovery,
        'will_pay':will_pay,'can_pay':can_pay,
    })

# ── VARIABLE DICTIONARY ───────────────────────────────────────────
VAR_DICT = [
    # Delinquency
    ("current_dpd","int","Delinquency Status","Current days past due (30/60/90/120/150/180)","90"),
    ("max_dpd_6m","int","Delinquency Status","Maximum DPD observed in last 6 months","120"),
    ("times_30dpd","int","Delinquency Status","Number of times account hit 30+ DPD in history","3"),
    ("times_60dpd","int","Delinquency Status","Number of times account hit 60+ DPD in history","2"),
    ("times_90dpd","int","Delinquency Status","Number of times account hit 90+ DPD in history","1"),
    ("dpd_at_last_statement","int","Delinquency Status","DPD as of last monthly statement","60"),
    ("mob","int","Delinquency Status","Months on book — account age","24"),
    ("days_since_first_delinq","int","Delinquency Status","Days since customer first became delinquent","90"),
    # Payment
    ("payment_ratio_3m","float","Payment Behaviour","Ratio of amount paid to amount due in last 3 months (0-1)","0.45"),
    ("payment_ratio_6m","float","Payment Behaviour","Ratio of amount paid to amount due in last 6 months (0-1)","0.38"),
    ("ptp_kept_rate","float","Payment Behaviour","Rate at which customer kept their Promise-to-Pay commitments","0.60"),
    ("ptp_broken_rate","float","Payment Behaviour","Rate at which customer broke their Promise-to-Pay commitments","0.40"),
    ("partial_payment_flag","int","Payment Behaviour","1 if customer made partial payment in last cycle","1"),
    ("last_payment_amount","int","Payment Behaviour","Amount paid in the most recent payment ($)","5000"),
    ("months_since_last_payment","int","Payment Behaviour","Months elapsed since last payment was made","2"),
    ("payment_frequency","str","Payment Behaviour","Payment pattern: Never/Irregular/Monthly/Bi-monthly","Monthly"),
    ("min_pay_count","int","Payment Behaviour","Number of months only minimum due was paid in last 6M","3"),
    ("full_pay_count","int","Payment Behaviour","Number of months full outstanding was paid in last 6M","0"),
    ("auto_debit_flag","int","Payment Behaviour","1 if auto-debit/ECS mandate is active","1"),
    ("payment_channel","str","Payment Behaviour","Last payment channel used: Online/Branch/UPI/NEFT/Cheque","UPI"),
    # Outstanding
    ("total_outstanding","int","Outstanding & Exposure","Total amount outstanding on the account ($)","75000"),
    ("principal_outstanding","int","Outstanding & Exposure","Principal component of outstanding ($)","50000"),
    ("interest_outstanding","int","Outstanding & Exposure","Interest and finance charges outstanding ($)","18000"),
    ("fees_outstanding","int","Outstanding & Exposure","Late fees and other charges outstanding ($)","7000"),
    ("credit_limit","int","Outstanding & Exposure","Sanctioned credit limit on the card ($)","100000"),
    ("utilization","float","Outstanding & Exposure","Credit utilization ratio (outstanding/limit), capped at 1","0.75"),
    ("overlimit_flag","int","Outstanding & Exposure","1 if outstanding exceeds credit limit","0"),
    ("revolving_balance","int","Outstanding & Exposure","Carried-forward balance not paid from previous cycles ($)","30000"),
    ("settlement_offer_pct","float","Outstanding & Exposure","Settlement offer as % of total outstanding (0.4–0.9)","0.70"),
    ("write_off_risk_score","float","Outstanding & Exposure","Internal score indicating write-off likelihood (0–100)","45.5"),
    # Bureau
    ("bureau_score","int","Bureau Signals","Credit bureau score (CIBIL/Experian equivalent, 300–800)","620"),
    ("bureau_score_change_3m","int","Bureau Signals","Change in bureau score over last 3 months (negative=drop)","−25"),
    ("num_active_loans","int","Bureau Signals","Number of active loan accounts across all lenders","3"),
    ("num_delinquent_accounts","int","Bureau Signals","Number of delinquent accounts across all lenders in bureau","2"),
    ("total_bureau_outstanding","int","Bureau Signals","Total outstanding across all credit facilities in bureau ($)","500000"),
    ("num_inquiries_3m","int","Bureau Signals","Number of credit inquiries in last 3 months","4"),
    ("secured_loan_flag","int","Bureau Signals","1 if customer has an active secured loan (home/auto)","1"),
    ("personal_loan_flag","int","Bureau Signals","1 if customer has an active personal loan","1"),
    ("home_loan_flag","int","Bureau Signals","1 if customer has an active home loan","0"),
    ("loan_to_income_ratio","float","Bureau Signals","Total EMI obligation as ratio of estimated monthly income","2.5"),
    ("bureau_dpd_max","int","Bureau Signals","Maximum DPD observed across all facilities in bureau","90"),
    ("credit_vintage","int","Bureau Signals","Months since oldest credit account was opened","48"),
    # Collections
    ("num_calls_made","int","Collections Activity","Total number of collection calls made to customer","8"),
    ("num_calls_answered","int","Collections Activity","Number of calls answered by customer","3"),
    ("rpc_flag","int","Collections Activity","Right Party Contact — 1 if customer was directly reached","1"),
    ("ptp_flag","int","Collections Activity","1 if customer gave a Promise-to-Pay during contact","1"),
    ("ptp_amount","int","Collections Activity","Amount promised to pay by customer ($, 0 if no PTP)","10000"),
    ("broken_ptp_count","int","Collections Activity","Number of times customer broke previous PTP commitments","2"),
    ("field_visit_flag","int","Collections Activity","1 if a field recovery agent was deployed","0"),
    ("legal_notice_flag","int","Collections Activity","1 if a legal notice has been issued","0"),
    ("settlement_offered_flag","int","Collections Activity","1 if a one-time settlement was formally offered","1"),
    ("last_contact_channel","str","Collections Activity","Channel of last contact: Call/SMS/Email/WhatsApp/Field","WhatsApp"),
    ("days_since_last_contact","int","Collections Activity","Days since last contact attempt was made","5"),
    ("contact_attempt_freq","int","Collections Activity","Average contact attempts per week in last month","3"),
    ("escalation_flag","int","Collections Activity","1 if account has been escalated to higher collection bucket","0"),
    ("agency_assigned_flag","int","Collections Activity","1 if account is assigned to external collection agency","0"),
    ("collection_stage","str","Collections Activity","Current stage: Early/Mid/Late/Pre-Writeoff","Mid"),
    # Transactions
    ("avg_spend_3m","float","Transaction Behaviour","Average monthly card spend in last 3 months ($)","15000"),
    ("spend_drop_flag","int","Transaction Behaviour","1 if spend in last 3M is significantly lower than previous 3M","1"),
    ("cash_advance_count_3m","int","Transaction Behaviour","Number of cash advance transactions in last 3 months","2"),
    ("declined_txn_count","int","Transaction Behaviour","Number of declined transactions in last month","3"),
    ("atm_withdrawal_3m","int","Transaction Behaviour","Total ATM cash withdrawals in last 3 months ($)","8000"),
    ("online_spend_ratio","float","Transaction Behaviour","Proportion of spend on online merchants (0–1)","0.65"),
    ("merchant_cat_diversity","int","Transaction Behaviour","Number of distinct merchant categories transacted","6"),
    ("spend_6m_vs_12m","float","Transaction Behaviour","Ratio of 6M spend to 12M spend — detects recent drop","0.75"),
    ("last_txn_days_ago","int","Transaction Behaviour","Days since last card transaction","15"),
    ("card_active_flag","int","Transaction Behaviour","1 if card was used at least once in last 30 days","1"),
    # Income
    ("salary_credit_flag","int","Income Proxies","1 if regular salary credits are seen in account","1"),
    ("avg_salary_credit_3m","int","Income Proxies","Average salary credit amount in last 3 months ($)","50000"),
    ("salary_credit_drop_flag","int","Income Proxies","1 if salary credit amount dropped >20% vs prior 3M","0"),
    ("employer_category","str","Income Proxies","Employer type: PSU/Private/Self-Employed/Govt/Gig","Private"),
    ("salary_credit_months_cnt","int","Income Proxies","Number of months salary was credited in last 12 months","10"),
    ("income_estimate","int","Income Proxies","Estimated annual income based on credits and bureau ($)","600000"),
    ("income_stability_score","float","Income Proxies","Score 0–100 measuring consistency of income inflows","72.5"),
    ("gig_worker_flag","int","Income Proxies","1 if income pattern suggests gig/freelance employment","0"),
    # Account
    ("months_on_book","int","Account Profile","Months since card was issued (same as MOB)","24"),
    ("product_type","str","Account Profile","Card product tier: Classic/Gold/Platinum/Signature","Gold"),
    ("card_variant","str","Account Profile","Card variant: Standard/Rewards/Cashback/Travel","Rewards"),
    ("acquisition_channel","str","Account Profile","How customer was acquired: Branch/DSA/Online/Referral","DSA"),
    ("relationship_banking_flag","int","Account Profile","1 if customer has a savings/current account with same bank","1"),
    ("num_products_with_bank","int","Account Profile","Total number of products held with the bank","3"),
    ("cross_sell_flag","int","Account Profile","1 if customer has been cross-sold a product recently","0"),
    ("premium_customer_flag","int","Account Profile","1 if customer is tagged as premium by bank CRM","0"),
    ("vintage_segment","str","Account Profile","Account age segment: New/Growing/Mature/Long-term","Mature"),
    ("customer_segment","str","Account Profile","Bank CRM segment: Mass/Affluent/HNI","Affluent"),
    # Contact
    ("city_tier","str","Contact & Reachability","City classification: Tier1/Tier2/Tier3","Tier2"),
    ("state","str","Contact & Reachability","State of customer residence","MH"),
    ("contact_score","float","Contact & Reachability","Propensity score for successful contact (0–100)","68.0"),
    ("email_valid_flag","int","Contact & Reachability","1 if email ID is valid and not bouncing","1"),
    ("mobile_valid_flag","int","Contact & Reachability","1 if mobile number is valid and reachable","1"),
    ("whatsapp_opt_in","int","Contact & Reachability","1 if customer has opted in for WhatsApp communications","1"),
    ("preferred_contact_time","str","Contact & Reachability","Best time to contact: Morning/Afternoon/Evening","Evening"),
    # Derived
    ("ability_score_raw","float","Derived / Engineered","Raw ability-to-pay score (0–1) based on income and exposure signals","0.62"),
    ("willingness_score_raw","float","Derived / Engineered","Raw willingness-to-pay score (0–1) based on behavioural signals","0.55"),
    ("roll_forward_risk","float","Derived / Engineered","Probability of rolling to next delinquency bucket (0–1)","0.35"),
    ("cure_probability_raw","float","Derived / Engineered","Raw probability of account curing (becoming current) (0–1)","0.48"),
    ("days_to_chargeoff_risk","int","Derived / Engineered","Estimated days until account reaches charge-off if no payment","90"),
    ("net_collectible_amount","int","Derived / Engineered","Expected collectible amount after settlement discount ($)","52500"),
    ("expected_recovery","int","Derived / Engineered","Expected recovery = net_collectible × cure_probability ($)","25200"),
    # Targets
    ("will_pay","int","TARGET","1 if customer made a payment within 30 days of collections contact","1"),
    ("can_pay","int","TARGET","1 if customer has financial capacity to pay (ability-based)","1"),
]

# ── MAIN APP ──────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3a5f,#2d6a9f);
                padding:32px;border-radius:12px;margin-bottom:24px'>
        <h1 style='color:white;margin:0;font-size:1.8rem'>
            💼 Capstone Project 1
        </h1>
        <h2 style='color:#93c5fd;margin:8px 0 0 0;font-size:1.2rem;font-weight:400'>
            Collections Intelligence: Willingness & Ability to Pay Modelling
        </h2>
        <p style='color:#bfdbfe;margin:8px 0 0 0;font-size:0.85rem'>
            Independent Project &nbsp;|&nbsp; 20,000 Customers &nbsp;|&nbsp;
            100 Variables &nbsp;|&nbsp; 2 Targets
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Problem Statement",
        "📖 Variable Dictionary",
        "👁️ Data Preview",
        "⬇️ Download Dataset"
    ])

    # ── TAB 1: PROBLEM STATEMENT ─────────────────────────────────
    with tab1:
        st.markdown("## 🏦 Business Context")
        st.markdown("""
When a credit card customer misses a payment, the bank's collections team must act — but resources
are finite. A collections team of 50 agents cannot call all 20,000 delinquent customers daily.
They need to **prioritise** — who to contact first, how aggressively, and what offer to make.

Two fundamental questions drive every collections decision:

> **"Can this customer pay if they wanted to?"** → Ability to Pay
>
> **"Will this customer pay if we contact them?"** → Willingness to Pay

The combination of these two assessments drives the entire collections strategy.
        """)

        st.markdown("## 🎯 Problem Statement")
        st.info("""
You are a **Credit Risk Data Scientist** at a mid-sized retail bank. The Collections Head has
approached you with a mandate:

*"Our recovery rate has stagnated at 34% for three quarters. We are spending ₹2.8Cr/month on
collections calls with poor targeting — agents are calling customers who will never pay, and
missing customers who would pay with a simple SMS reminder. Build me a model that tells my team
who to call, in what order, and with what offer."*

**Your objective:**
1. Build a **Willingness-to-Pay model** — predict which customers will respond to collections
   contact and make a payment within 30 days (`will_pay = 1`)
2. Build an **Ability-to-Pay model** — predict which customers have the financial capacity
   to pay (`can_pay = 1`)
3. Combine both models into a **2×2 segmentation matrix** to drive collections strategy
        """)

        st.markdown("## 📊 Dataset Description")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", "20,000")
        col2.metric("Variables", "100")
        col3.metric("Will Pay Rate", "~70%")
        col4.metric("Can Pay Rate", "~75%")

        st.markdown("""
**Customer Universe:** Credit card customers with at least one missed payment (30+ DPD)

**Variable Groups:**
| Group | Count | Description |
|---|---|---|
| Delinquency Status | 8 | DPD history, MOB, delinquency timeline |
| Payment Behaviour | 12 | Payment ratios, PTP history, channel |
| Outstanding & Exposure | 10 | Balances, limits, utilization |
| Bureau Signals | 12 | Credit score, multi-lender exposure |
| Collections Activity | 15 | Calls, PTP, field visits, stage |
| Transaction Behaviour | 10 | Spend, cash advances, card activity |
| Income Proxies | 8 | Salary credits, employer, stability |
| Account Profile | 10 | MOB, product, segment |
| Contact & Reachability | 7 | City tier, contact score, opt-ins |
| Derived / Engineered | 7 | Pre-built scores and risk signals |
| **Targets** | **2** | **will_pay, can_pay** |
        """)

        st.markdown("## 🎯 Expected Deliverables")
        st.success("""
Your submission should include:

**1. Exploratory Data Analysis**
- Distribution of targets (will_pay, can_pay)
- Key variable correlations with each target
- Missing value analysis and treatment strategy
- Outlier detection and treatment

**2. Feature Engineering**
- Handle categorical variables (encoding)
- Create interaction features if relevant
- Variable selection / reduction (variance, correlation, importance)

**3. Willingness-to-Pay Model**
- Logistic Regression baseline
- XGBoost / Random Forest
- Evaluation: AUC-ROC, KS Statistic, Gini Coefficient
- SHAP feature importance — top 10 drivers of willingness

**4. Ability-to-Pay Model**
- Same modelling approach as willingness
- Evaluate separately

**5. 2×2 Collections Strategy Matrix**
- Segment all 20,000 customers into 4 quadrants:
  - High Ability + High Willingness → Self-cure, SMS only
  - High Ability + Low Willingness → Intensive calling, legal notice
  - Low Ability + High Willingness → Restructure / EMI offer
  - Low Ability + Low Willingness → Settle or write-off
- Show size and expected recovery per quadrant

**6. Business Recommendation**
- Recommended contact priority list
- Optimal threshold for each model
- Expected improvement in recovery rate vs current 34%
        """)

        st.markdown("## 📏 Evaluation Criteria")
        st.markdown("""
| Criterion | Weight |
|---|---|
| EDA quality & insights | 20% |
| Feature engineering & variable selection | 15% |
| Model performance (AUC, KS, Gini) | 25% |
| SHAP explainability | 15% |
| 2×2 matrix & business recommendation | 20% |
| Code quality & documentation | 5% |
        """)

        st.markdown("## 📚 Key Metrics Reference")
        with st.expander("What is KS Statistic?"):
            st.markdown("""
**Kolmogorov-Smirnov (KS) Statistic** measures the maximum separation between the cumulative
distribution of events (defaulters/payers) and non-events.
- KS > 40% = Good model
- KS > 60% = Very good model
- Used extensively in credit risk alongside AUC
            """)
        with st.expander("What is Gini Coefficient?"):
            st.markdown("""
**Gini = 2×AUC - 1**
- Gini of 0.6 = AUC of 0.80
- Gini > 0.5 is considered good for collections models
- Normalised measure of model discrimination
            """)
        with st.expander("What is the 2×2 Collections Matrix?"):
            st.markdown("""
| | **High Ability** | **Low Ability** |
|---|---|---|
| **High Willingness** | ✅ Self-cure — SMS reminder only | 🔄 Restructure / EMI plan |
| **Low Willingness** | 📞 Intensive calling + legal notice | ⚠️ OTS or write-off |

This matrix is the **core output** of any collections analytics project.
It tells the collections head exactly how to allocate their agent bandwidth.
            """)

    # ── TAB 2: VARIABLE DICTIONARY ───────────────────────────────
    with tab2:
        st.markdown("## 📖 Variable Dictionary")
        st.caption("100 variables + customer_id + 2 targets. Search by name, group or description.")

        search = st.text_input("🔍 Search variables", placeholder="e.g. ptp, bureau, salary...")
        group_filter = st.selectbox("Filter by group", ["All Groups"] + sorted(set(v[2] for v in VAR_DICT)))

        filtered = VAR_DICT
        if search:
            filtered = [v for v in filtered if search.lower() in v[0].lower()
                        or search.lower() in v[3].lower()]
        if group_filter != "All Groups":
            filtered = [v for v in filtered if v[2] == group_filter]

        vdf = pd.DataFrame(filtered,
                           columns=["Variable Name","Type","Group","Description","Example Value"])

        # Colour-code targets
        def highlight(row):
            if row["Group"] == "TARGET":
                return ["background-color:#fef9c3"]*len(row)
            return [""]*len(row)

        st.dataframe(
            vdf.style.apply(highlight, axis=1),
            use_container_width=True,
            height=600,
        )
        st.caption(f"Showing {len(filtered)} of {len(VAR_DICT)} variables")

    # ── TAB 3: DATA PREVIEW ──────────────────────────────────────
    with tab3:
        st.markdown("## 👁️ Dataset Preview — First 50 Rows")
        with st.spinner("Generating dataset..."):
            df = generate_dataset()
        st.dataframe(df.head(50), use_container_width=True, height=420)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Will Pay Rate", f"{round(df['will_pay'].mean()*100,1)}%")
        c4.metric("Can Pay Rate", f"{round(df['can_pay'].mean()*100,1)}%")

    # ── TAB 4: DOWNLOAD ──────────────────────────────────────────
    with tab4:
        st.markdown("## ⬇️ Download Dataset")
        st.info("Enter your details below to download. Your access will be logged.")
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
                label="📥 Click here to download collections_dataset.csv",
                data=csv,
                file_name="capstone1_collections_dataset.csv",
                mime="text/csv",
            )
            st.success(f"✅ Download ready! Logged for {email}")
            st.markdown(f"""
**Dataset details:**
- Rows: 20,000 customers
- Columns: {len(df.columns)} (100 variables + customer_id + will_pay + can_pay)
- Size: ~8 MB
- Format: CSV, UTF-8

**Targets:**
- `will_pay`: 1 if customer paid within 30 days of collections contact
- `can_pay`: 1 if customer has financial capacity to pay

Good luck with the project! 🚀
            """)

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
