"""
questions_m4.py — Module 4: Clustering & Dimensionality Reduction
3 Practice sets + 1 Final assessment (moderate level, 90 min, 1 attempt)
Topics: KMeans, Hierarchical, DBSCAN, PCA, t-SNE, UMAP
Practice Dataset: Banking customers (seed=303, 400 rows)
Final Dataset: Credit card customers (seed=404, 500 rows)
"""

import hashlib, re, io, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()

def _exec(code, globs):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:    exec(code, globs)
    except Exception as e: sys.stdout = old; return f"ERROR: {e}"
    sys.stdout = old
    return buf.getvalue().strip()

# ════════════════════════════════════════════════════════════════
# PRACTICE DATASET (seed=303, 400 rows)
# ════════════════════════════════════════════════════════════════
np.random.seed(303)
_np = 400
_df_p = pd.DataFrame({
    'customer_id':       range(7001, 7401),
    'age':               np.random.randint(22, 65, _np),
    'income':            np.random.randint(200000, 3000000, _np),
    'credit_score':      np.random.randint(450, 850, _np),
    'num_products':      np.random.randint(1, 6, _np),
    'months_with_bank':  np.random.randint(6, 120, _np),
    'credit_utilization':np.round(np.random.uniform(0.1, 0.95, _np), 2),
    'loan_amount':       np.random.randint(50000, 2000000, _np),
    'is_defaulter':      np.random.choice([0, 1], _np, p=[0.78, 0.22]),
})
_GP = {
    'df': _df_p.copy(), 'pd': pd, 'np': np,
    'KMeans': KMeans, 'PCA': PCA, 'StandardScaler': StandardScaler,
    'DBSCAN': DBSCAN, 'AgglomerativeClustering': AgglomerativeClustering,
    'silhouette_score': silhouette_score,
}

PRACTICE_DF_SETUP = """\
import pandas as pd, numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(303)
_n = 400
df = pd.DataFrame({
    'customer_id':        range(7001, 7401),
    'age':                np.random.randint(22, 65, _n),
    'income':             np.random.randint(200000, 3000000, _n),
    'credit_score':       np.random.randint(450, 850, _n),
    'num_products':       np.random.randint(1, 6, _n),
    'months_with_bank':   np.random.randint(6, 120, _n),
    'credit_utilization': np.round(np.random.uniform(0.1, 0.95, _n), 2),
    'loan_amount':        np.random.randint(50000, 2000000, _n),
    'is_defaulter':       np.random.choice([0, 1], _n, p=[0.78, 0.22]),
})
"""

# ════════════════════════════════════════════════════════════════
# FINAL DATASET (seed=404, 500 rows)
# ════════════════════════════════════════════════════════════════
np.random.seed(404)
_nf = 500
_df_f = pd.DataFrame({
    'customer_id':        range(8001, 8501),
    'age':                np.random.randint(22, 70, _nf),
    'annual_income':      np.random.randint(150000, 5000000, _nf),
    'credit_score':       np.random.randint(400, 900, _nf),
    'credit_limit':       np.random.randint(50000, 2000000, _nf),
    'credit_utilization': np.round(np.random.uniform(0.05, 0.98, _nf), 2),
    'num_credit_cards':   np.random.randint(1, 8, _nf),
    'months_with_bank':   np.random.randint(3, 180, _nf),
    'num_products':       np.random.randint(1, 6, _nf),
    'loan_amount':        np.random.randint(0, 3000000, _nf),
    'is_defaulter':       np.random.choice([0, 1], _nf, p=[0.80, 0.20]),
})
_GF = {
    'df': _df_f.copy(), 'pd': pd, 'np': np,
    'KMeans': KMeans, 'PCA': PCA, 'StandardScaler': StandardScaler,
    'silhouette_score': silhouette_score,
}
_FEAT_F = ['annual_income','credit_score','credit_limit','credit_utilization',
           'num_credit_cards','months_with_bank','num_products','loan_amount']

FINAL_DF_SETUP = """\
import pandas as pd, numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(404)
_n = 500
df = pd.DataFrame({
    'customer_id':        range(8001, 8501),
    'age':                np.random.randint(22, 70, _n),
    'annual_income':      np.random.randint(150000, 5000000, _n),
    'credit_score':       np.random.randint(400, 900, _n),
    'credit_limit':       np.random.randint(50000, 2000000, _n),
    'credit_utilization': np.round(np.random.uniform(0.05, 0.98, _n), 2),
    'num_credit_cards':   np.random.randint(1, 8, _n),
    'months_with_bank':   np.random.randint(3, 180, _n),
    'num_products':       np.random.randint(1, 6, _n),
    'loan_amount':        np.random.randint(0, 3000000, _n),
    'is_defaulter':       np.random.choice([0, 1], _n, p=[0.80, 0.20]),
})
FEATURES = ['annual_income','credit_score','credit_limit','credit_utilization',
            'num_credit_cards','months_with_bank','num_products','loan_amount']
"""

# ── PRECOMPUTE PRACTICE ANSWERS ──────────────────────────────────
_p1e4 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
km=KMeans(n_clusters=3,random_state=42,n_init=10)
df2=df.copy(); df2['cluster']=km.fit_predict(X)
print(df2['cluster'].value_counts().to_dict())
""", {**_GP, 'df': _df_p.copy()})

_p1e5 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X)
print(round(silhouette_score(X,labels),2))
""", {**_GP, 'df': _df_p.copy()})

_p1e7 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
inertias=[]
for k in range(2,7):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    km.fit(X)
    inertias.append(round(km.inertia_,2))
print(inertias)
""", {**_GP, 'df': _df_p.copy()})

_p1e8 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
km=KMeans(n_clusters=4,random_state=42,n_init=10)
df2=df.copy(); df2['cluster']=km.fit_predict(X)
print(df2.groupby('cluster')['income'].mean().round(0).to_dict())
""", {**_GP, 'df': _df_p.copy()})

_p2e3 = _exec("""
features=['income','credit_score','credit_utilization','loan_amount','months_with_bank']
X=StandardScaler().fit_transform(df[features])
pca=PCA(n_components=2,random_state=42)
print(pca.fit_transform(X).shape)
""", {**_GP, 'df': _df_p.copy()})

_p2e4 = _exec("""
features=['income','credit_score','credit_utilization','loan_amount','months_with_bank']
X=StandardScaler().fit_transform(df[features])
pca=PCA(n_components=2,random_state=42)
pca.fit(X)
print(round(pca.explained_variance_ratio_.sum()*100,1))
""", {**_GP, 'df': _df_p.copy()})

_p2e5 = _exec("""
features=['income','credit_score','credit_utilization','loan_amount','months_with_bank']
X=StandardScaler().fit_transform(df[features])
pca=PCA(random_state=42); pca.fit(X)
cumvar=0; nc=0
for v in pca.explained_variance_ratio_:
    cumvar+=v; nc+=1
    if cumvar>=0.90: break
print(nc)
""", {**_GP, 'df': _df_p.copy()})

_p2e7 = _exec("""
features=['income','credit_score','credit_utilization','loan_amount','months_with_bank']
X=StandardScaler().fit_transform(df[features])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
print(round(silhouette_score(X_pca,labels),2))
""", {**_GP, 'df': _df_p.copy()})

_p2e8 = _exec("""
features=['income','credit_score','credit_utilization','loan_amount','months_with_bank']
X=StandardScaler().fit_transform(df[features])
pca=PCA(n_components=3,random_state=42); pca.fit(X)
idx=pca.components_[0].argmax()
print(features[idx])
""", {**_GP, 'df': _df_p.copy(), 'features': ['income','credit_score','credit_utilization','loan_amount','months_with_bank']})

_p3e3 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
db=DBSCAN(eps=0.5,min_samples=5)
labels=db.fit_predict(X)
print(len(set(labels))-(1 if -1 in labels else 0))
""", {**_GP, 'df': _df_p.copy()})

_p3e4 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
db=DBSCAN(eps=0.5,min_samples=5)
labels=db.fit_predict(X)
print(sum(labels==-1))
""", {**_GP, 'df': _df_p.copy()})

_p3e5 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
agg=AgglomerativeClustering(n_clusters=3)
labels=agg.fit_predict(X)
print(pd.Series(labels).value_counts().to_dict())
""", {**_GP, 'df': _df_p.copy()})

_p3e7 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
db=DBSCAN(eps=0.8,min_samples=5)
labels=db.fit_predict(X)
print(len(set(labels))-(1 if -1 in labels else 0), sum(labels==-1))
""", {**_GP, 'df': _df_p.copy()})

_p3e8 = _exec("""
features=['income','credit_score','credit_utilization']
X=StandardScaler().fit_transform(df[features])
best_k,best_s=0,0
for k in range(2,7):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(X)
    s=silhouette_score(X,labels)
    if s>best_s: best_s=s; best_k=k
print(best_k, round(best_s,2))
""", {**_GP, 'df': _df_p.copy()})

# ── PRECOMPUTE FINAL ANSWERS ──────────────────────────────────────
_fe1  = "(500, 8)"
_fe2  = "(500, 2)"
_fe3  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
pca=PCA(n_components=2,random_state=42); pca.fit(X)
print(round(pca.explained_variance_ratio_.sum()*100,1))
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe4  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
pca=PCA(random_state=42); pca.fit(X)
cumvar=0; nc=0
for v in pca.explained_variance_ratio_:
    cumvar+=v; nc+=1
    if cumvar>=0.80: break
print(nc)
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe5  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
df2=df.copy(); df2['cluster']=labels
print(df2['cluster'].value_counts().to_dict())
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe6  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
print(round(silhouette_score(X_pca,labels),2))
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe7  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
df2=df.copy(); df2['cluster']=labels
print(df2.groupby('cluster')['annual_income'].mean().round(0).to_dict())
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe8  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
df2=df.copy(); df2['cluster']=labels
print(df2.groupby('cluster')['is_defaulter'].mean().round(2).to_dict())
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe9  = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
pca=PCA(n_components=2,random_state=42); pca.fit(X)
idx=pca.components_[0].argmax()
print(FEATURES[idx])
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe10 = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
best_k,best_s=2,0
for k in range(2,7):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(X_pca)
    s=silhouette_score(X_pca,labels)
    if s>best_s: best_s=s; best_k=k
print(best_k, round(best_s,2))
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe11 = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
df2=df.copy(); df2['cluster']=labels
print(df2.groupby('cluster')['credit_score'].mean().round(0).to_dict())
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

_fe12 = _exec("""
X=StandardScaler().fit_transform(df[FEATURES])
X_pca=PCA(n_components=2,random_state=42).fit_transform(X)
km=KMeans(n_clusters=3,random_state=42,n_init=10)
labels=km.fit_predict(X_pca)
df2=df.copy(); df2['cluster']=labels
print(df2.groupby('cluster')['credit_utilization'].mean().idxmax())
""", {**_GF,'df':_df_f.copy(),'FEATURES':_FEAT_F})

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 1 — KMeans Clustering
# ════════════════════════════════════════════════════════════════
M4_PRACTICE_1 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"KMeans clustering requires you to specify K upfront. What does K represent?",
     "opts":["Number of features","Number of data points","Number of clusters","Number of iterations"],
     "ah":h("Number of clusters"),
     "solution":"# K = number of clusters to form\nkm = KMeans(n_clusters=3, random_state=42)\nkm.fit(X)\n# km.labels_ gives cluster assignment for each point",
     "explanation":"K defines how many groups KMeans partitions data into. Choosing the right K is a key challenge — use elbow method or silhouette score."},
    {"id":2,"type":"mcq","marks":2,
     "text":"Why must features be scaled before applying KMeans?",
     "opts":[
         "KMeans cannot handle float values",
         "KMeans uses distance — unscaled features with larger ranges dominate the clustering",
         "Scaling makes the algorithm faster only",
         "Scaling is required only for PCA, not KMeans"
     ],
     "ah":h("KMeans uses distance — unscaled features with larger ranges dominate the clustering"),
     "solution":"from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(df[features])\n# Now income (millions) won't dominate credit_score (hundreds)",
     "explanation":"KMeans minimizes Euclidean distance. A feature with range [0,1M] will dominate one with range [0,1]. StandardScaler brings all features to mean=0, std=1."},
    {"id":3,"type":"mcq","marks":2,
     "text":"What does the inertia (within-cluster sum of squares) measure in KMeans?",
     "opts":[
         "Distance between cluster centers",
         "Sum of squared distances from each point to its cluster center",
         "Number of outliers per cluster",
         "Total variance in the dataset"
     ],
     "ah":h("Sum of squared distances from each point to its cluster center"),
     "solution":"km = KMeans(n_clusters=3, random_state=42)\nkm.fit(X_scaled)\nprint(km.inertia_)  # lower = tighter clusters",
     "explanation":"Inertia measures compactness. Lower inertia = tighter clusters. Used in the elbow method — plot inertia vs K and find the elbow."},
    {"id":4,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\nAll sklearn imports available.\n\nScale these 3 features with StandardScaler.\nApply KMeans with n_clusters=3, random_state=42, n_init=10.\nPrint the cluster size distribution as a dict.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p1e4,
     "solution":f"features = ['income', 'credit_score', 'credit_utilization']\nX = StandardScaler().fit_transform(df[features])\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X)\nprint(df2['cluster'].value_counts().to_dict())\n# {_p1e4}",
     "explanation":"fit_predict() scales, fits and assigns labels in one step. value_counts() shows how many points in each cluster."},
    {"id":5,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features, apply KMeans (n_clusters=3, random_state=42, n_init=10).\nCompute the silhouette score.\nPrint rounded to 2 decimal places.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p1e5,
     "solution":f"features = ['income', 'credit_score', 'credit_utilization']\nX = StandardScaler().fit_transform(df[features])\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\nlabels = km.fit_predict(X)\nprint(round(silhouette_score(X, labels), 2))\n# {_p1e5}",
     "explanation":"Silhouette score ranges from -1 to 1. Higher = better defined clusters. Values around 0.25-0.5 are common for real data."},
    {"id":6,"type":"mcq","marks":3,
     "text":"In the elbow method for choosing K, what does the 'elbow' represent?",
     "opts":[
         "The K value where inertia becomes zero",
         "The K value where adding more clusters gives diminishing returns in reducing inertia",
         "The K value with the highest silhouette score",
         "The K value where all clusters have equal size"
     ],
     "ah":h("The K value where adding more clusters gives diminishing returns in reducing inertia"),
     "solution":"inertias = []\nfor k in range(2, 8):\n    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n    km.fit(X_scaled)\n    inertias.append(km.inertia_)\n# Plot inertias vs k — the 'elbow' is where the curve bends",
     "explanation":"Inertia always decreases as K increases. The elbow is the point of diminishing returns — beyond it, adding clusters doesn't improve compactness much."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nCompute KMeans inertia for K = 2, 3, 4, 5, 6.\nPrint as a list of inertia values rounded to 2 decimal places.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p1e7,
     "solution":f"features = ['income', 'credit_score', 'credit_utilization']\nX = StandardScaler().fit_transform(df[features])\ninertias = []\nfor k in range(2, 7):\n    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n    km.fit(X)\n    inertias.append(round(km.inertia_, 2))\nprint(inertias)\n# {_p1e7}",
     "explanation":"Inertia should decrease as K increases. Plot these values to find the elbow — the K where the drop starts to flatten."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nApply KMeans with n_clusters=4, random_state=42, n_init=10.\nPrint mean income per cluster as a dict (rounded to 0 decimal places).",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p1e8,
     "solution":f"features = ['income', 'credit_score', 'credit_utilization']\nX = StandardScaler().fit_transform(df[features])\nkm = KMeans(n_clusters=4, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X)\nprint(df2.groupby('cluster')['income'].mean().round(0).to_dict())\n# {_p1e8}",
     "explanation":"Profiling clusters by original features helps interpret them. Clusters with high mean income represent premium customer segments."},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 2 — PCA & Dimensionality Reduction
# ════════════════════════════════════════════════════════════════
M4_PRACTICE_2 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What is the primary goal of Principal Component Analysis (PCA)?",
     "opts":[
         "Remove outliers from the dataset",
         "Reduce dimensionality while preserving maximum variance",
         "Normalize features to [0,1] range",
         "Cluster data into groups"
     ],
     "ah":h("Reduce dimensionality while preserving maximum variance"),
     "solution":"from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n# Step 1: Scale features\nX_scaled = StandardScaler().fit_transform(X)\n# Step 2: Apply PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\n# 5 features → 2 principal components",
     "explanation":"PCA finds directions of maximum variance. By projecting onto fewer components, you reduce dimensions while keeping most information."},
    {"id":2,"type":"mcq","marks":2,
     "text":"What does explained_variance_ratio_ tell you in PCA?",
     "opts":[
         "The number of components needed",
         "The proportion of total variance captured by each principal component",
         "The correlation between components",
         "The eigenvalues of each feature"
     ],
     "ah":h("The proportion of total variance captured by each principal component"),
     "solution":"pca = PCA(n_components=3)\npca.fit(X_scaled)\nprint(pca.explained_variance_ratio_)\n# [0.45, 0.25, 0.15] → PC1 captures 45%, PC2 captures 25%, etc.\nprint(pca.explained_variance_ratio_.sum())  # total = 0.85 = 85%",
     "explanation":"explained_variance_ratio_ sums to 1 across all components. Use cumulative sum to find how many components are needed to explain X% of variance."},
    {"id":3,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization, loan_amount, months_with_bank\n\nScale features with StandardScaler.\nApply PCA with n_components=2, random_state=42.\nPrint the shape of the resulting array.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p2e3,
     "solution":f"features = ['income','credit_score','credit_utilization','loan_amount','months_with_bank']\nX = StandardScaler().fit_transform(df[features])\npca = PCA(n_components=2, random_state=42)\nX_pca = pca.fit_transform(X)\nprint(X_pca.shape)\n# {_p2e3}",
     "explanation":f"5 features reduced to 2 components. 400 rows preserved. Output shape = {_p2e3}."},
    {"id":4,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization, loan_amount, months_with_bank\n\nScale features, apply PCA with n_components=2, random_state=42.\nPrint the total explained variance (%) rounded to 1 decimal place.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p2e4,
     "solution":f"features = ['income','credit_score','credit_utilization','loan_amount','months_with_bank']\nX = StandardScaler().fit_transform(df[features])\npca = PCA(n_components=2, random_state=42)\npca.fit(X)\nprint(round(pca.explained_variance_ratio_.sum()*100, 1))\n# {_p2e4}%",
     "explanation":f"2 components capture {_p2e4}% of total variance. The remaining {round(100-float(_p2e4),1)}% is lost in the dimensionality reduction."},
    {"id":5,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization, loan_amount, months_with_bank\n\nScale features, fit PCA on all components.\nFind the minimum number of components needed to explain 90% of variance.\nPrint that number.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p2e5,
     "solution":f"features = ['income','credit_score','credit_utilization','loan_amount','months_with_bank']\nX = StandardScaler().fit_transform(df[features])\npca = PCA(random_state=42)\npca.fit(X)\ncumvar = 0; nc = 0\nfor v in pca.explained_variance_ratio_:\n    cumvar += v; nc += 1\n    if cumvar >= 0.90: break\nprint(nc)\n# {_p2e5}",
     "explanation":f"Iterating through explained_variance_ratio_ and accumulating until reaching 0.90. Answer = {_p2e5} components."},
    {"id":6,"type":"mcq","marks":3,
     "text":"What is the key difference between t-SNE and PCA for dimensionality reduction?",
     "opts":[
         "t-SNE preserves global variance; PCA preserves local structure",
         "PCA is linear and preserves global variance; t-SNE is non-linear and preserves local neighborhood structure",
         "t-SNE requires more components than PCA",
         "PCA cannot reduce to 2D; t-SNE can"
     ],
     "ah":h("PCA is linear and preserves global variance; t-SNE is non-linear and preserves local neighborhood structure"),
     "solution":"# PCA: linear, deterministic, preserves global variance\n# Use for: preprocessing, feature reduction before ML\n# t-SNE: non-linear, stochastic, good for visualization\n# Use for: 2D/3D visualization of high-dimensional data\n# UMAP: similar to t-SNE but faster and better global structure",
     "explanation":"PCA is a linear transformation best for preprocessing. t-SNE is non-linear, great for 2D visualization of clusters but results vary with random seed and perplexity."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization, loan_amount, months_with_bank\n\nReduce to 2 PCA components (StandardScaler first, random_state=42).\nApply KMeans (n_clusters=3, random_state=42, n_init=10) on PCA output.\nPrint silhouette score rounded to 2 decimal places.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p2e7,
     "solution":f"features = ['income','credit_score','credit_utilization','loan_amount','months_with_bank']\nX = StandardScaler().fit_transform(df[features])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\nlabels = km.fit_predict(X_pca)\nprint(round(silhouette_score(X_pca, labels), 2))\n# {_p2e7}",
     "explanation":f"KMeans on PCA-reduced data. Silhouette = {_p2e7}. PCA can improve clustering by removing noise dimensions."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization, loan_amount, months_with_bank\n\nScale features, apply PCA with n_components=3, random_state=42.\nFind which original feature contributes most to PC1 (first component).\nPrint only the feature name.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p2e8,
     "solution":f"features = ['income','credit_score','credit_utilization','loan_amount','months_with_bank']\nX = StandardScaler().fit_transform(df[features])\npca = PCA(n_components=3, random_state=42)\npca.fit(X)\nidx = pca.components_[0].argmax()\nprint(features[idx])\n# {_p2e8}",
     "explanation":f"pca.components_[0] contains loadings for PC1. argmax() finds the feature with the highest (most positive) contribution. Answer: {_p2e8}."},
]

# ════════════════════════════════════════════════════════════════
# PRACTICE SET 3 — DBSCAN, Hierarchical & Comparison
# ════════════════════════════════════════════════════════════════
M4_PRACTICE_3 = [
    {"id":1,"type":"mcq","marks":2,
     "text":"What is the key advantage of DBSCAN over KMeans?",
     "opts":[
         "DBSCAN is always faster than KMeans",
         "DBSCAN can find arbitrarily shaped clusters and automatically detect outliers",
         "DBSCAN requires you to specify the number of clusters upfront",
         "DBSCAN always produces exactly 3 clusters"
     ],
     "ah":h("DBSCAN can find arbitrarily shaped clusters and automatically detect outliers"),
     "solution":"# DBSCAN: density-based, no need to specify K\n# Points with label=-1 are noise/outliers\n# Can find non-spherical clusters\ndb = DBSCAN(eps=0.5, min_samples=5)\nlabels = db.fit_predict(X_scaled)\nnoise = sum(labels == -1)",
     "explanation":"DBSCAN groups dense regions. Points in sparse regions get label=-1 (noise/outlier). It doesn't assume spherical clusters like KMeans."},
    {"id":2,"type":"mcq","marks":2,
     "text":"In DBSCAN, what do the parameters eps and min_samples control?",
     "opts":[
         "eps = number of clusters; min_samples = minimum cluster size",
         "eps = neighbourhood radius; min_samples = minimum points to form a dense region",
         "eps = distance threshold for outliers; min_samples = total dataset size",
         "eps = learning rate; min_samples = number of iterations"
     ],
     "ah":h("eps = neighbourhood radius; min_samples = minimum points to form a dense region"),
     "solution":"# eps: if two points are within eps distance, they're neighbours\n# min_samples: a core point needs at least min_samples neighbours\n# Larger eps → fewer, bigger clusters\n# Smaller min_samples → more core points",
     "explanation":"eps controls how close points must be to be neighbours. min_samples controls how dense a region must be to start a cluster."},
    {"id":3,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features. Apply DBSCAN with eps=0.5, min_samples=5.\nPrint the number of clusters found (excluding noise label -1).",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p3e3,
     "solution":f"features = ['income','credit_score','credit_utilization']\nX = StandardScaler().fit_transform(df[features])\ndb = DBSCAN(eps=0.5, min_samples=5)\nlabels = db.fit_predict(X)\nn_clusters = len(set(labels)) - (1 if -1 in labels else 0)\nprint(n_clusters)\n# {_p3e3}",
     "explanation":f"set(labels) gives unique cluster ids including -1 (noise). Subtract 1 if -1 is present. Answer = {_p3e3} clusters."},
    {"id":4,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features. Apply DBSCAN with eps=0.5, min_samples=5.\nPrint the number of noise points (label = -1).",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p3e4,
     "solution":f"features = ['income','credit_score','credit_utilization']\nX = StandardScaler().fit_transform(df[features])\ndb = DBSCAN(eps=0.5, min_samples=5)\nlabels = db.fit_predict(X)\nprint(sum(labels == -1))\n# {_p3e4}",
     "explanation":f"DBSCAN assigns -1 to points that don't belong to any dense cluster. {_p3e4} noise points detected with eps=0.5."},
    {"id":5,"type":"code","marks":3,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features. Apply AgglomerativeClustering with n_clusters=3.\nPrint cluster size distribution as a dict.",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p3e5,
     "solution":f"features = ['income','credit_score','credit_utilization']\nX = StandardScaler().fit_transform(df[features])\nagg = AgglomerativeClustering(n_clusters=3)\nlabels = agg.fit_predict(X)\nprint(pd.Series(labels).value_counts().to_dict())\n# {_p3e5}",
     "explanation":f"Hierarchical clustering merges closest clusters bottom-up. Result: {_p3e5}. No random_state needed — deterministic algorithm."},
    {"id":6,"type":"mcq","marks":3,
     "text":"When would you prefer Hierarchical clustering over KMeans?",
     "opts":[
         "When you have millions of data points",
         "When you need to explore multiple K values without rerunning and want a dendrogram",
         "When all clusters must be equal size",
         "When features are categorical"
     ],
     "ah":h("When you need to explore multiple K values without rerunning and want a dendrogram"),
     "solution":"# Hierarchical: builds a full dendrogram\n# Cut at different heights to get different K values\n# Computationally expensive O(n²) — not for large datasets\n# Best for: small datasets, exploratory analysis",
     "explanation":"The dendrogram shows the merge history. You can pick any K by cutting at different heights without rerunning the algorithm. But O(n²) complexity limits it to small datasets."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features. Apply DBSCAN with eps=0.8, min_samples=5.\nPrint: number of clusters and number of noise points (space separated).",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p3e7,
     "solution":f"features = ['income','credit_score','credit_utilization']\nX = StandardScaler().fit_transform(df[features])\ndb = DBSCAN(eps=0.8, min_samples=5)\nlabels = db.fit_predict(X)\nn_clusters = len(set(labels)) - (1 if -1 in labels else 0)\nn_noise = sum(labels == -1)\nprint(n_clusters, n_noise)\n# {_p3e7}",
     "explanation":f"Increasing eps from 0.5 to 0.8 expands neighborhoods — more points qualify as core points, reducing noise. Result: {_p3e7}."},
    {"id":8,"type":"code","marks":4,
     "text":"df is preloaded (400 rows).\nFeatures: income, credit_score, credit_utilization\n\nScale features. Find the best K for KMeans (K from 2 to 6) using silhouette score.\nPrint: best_k and best_silhouette_score (space separated, score rounded to 2 decimal places).",
     "preload":PRACTICE_DF_SETUP,
     "exp":_p3e8,
     "solution":f"features = ['income','credit_score','credit_utilization']\nX = StandardScaler().fit_transform(df[features])\nbest_k, best_s = 0, 0\nfor k in range(2, 7):\n    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n    labels = km.fit_predict(X)\n    s = silhouette_score(X, labels)\n    if s > best_s:\n        best_s = s; best_k = k\nprint(best_k, round(best_s, 2))\n# {_p3e8}",
     "explanation":f"Iterate K=2..6, compute silhouette for each, track best. Best K = {_p3e8.split()[0]} with silhouette = {_p3e8.split()[1]}."},
]

# ════════════════════════════════════════════════════════════════
# FINAL ASSESSMENT — KMeans + PCA on Credit Dataset
# 90 minutes, 1 attempt, moderate level
# ════════════════════════════════════════════════════════════════
M4_FINAL = [
    {"id":1,"type":"mcq","marks":3,
     "text":"Before applying PCA, you must scale your features. Which scaler is most appropriate and why?",
     "opts":[
         "MinMaxScaler — compresses all features to [0,1]",
         "StandardScaler — PCA is sensitive to variance; StandardScaler makes all features contribute equally",
         "No scaling needed — PCA handles it internally",
         "RobustScaler — only suitable option for PCA"
     ],
     "ah":h("StandardScaler — PCA is sensitive to variance; StandardScaler makes all features contribute equally"),
     "solution":"from sklearn.preprocessing import StandardScaler\n# PCA finds directions of maximum variance\n# Without scaling, high-variance features (e.g. income) dominate\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(df[FEATURES])\n# Now all features have mean=0, std=1",
     "explanation":"PCA maximises variance. Without scaling, income (range: millions) dominates credit_score (range: 400-900). StandardScaler equalises feature contributions."},
    {"id":2,"type":"code","marks":3,
     "text":"df is preloaded (500 rows).\nFEATURES = ['annual_income','credit_score','credit_limit','credit_utilization','num_credit_cards','months_with_bank','num_products','loan_amount']\n\nScale FEATURES with StandardScaler.\nPrint the shape of the scaled array.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe1,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nprint(X.shape)\n# {_fe1}",
     "explanation":f"500 rows × 8 features after scaling. Shape = {_fe1}. Scaling preserves shape — only values change."},
    {"id":3,"type":"code","marks":3,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nScale FEATURES with StandardScaler.\nApply PCA with n_components=2, random_state=42.\nPrint the total variance explained (%) rounded to 1 decimal place.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe3,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\npca = PCA(n_components=2, random_state=42)\npca.fit(X)\nprint(round(pca.explained_variance_ratio_.sum()*100, 1))\n# {_fe3}%",
     "explanation":f"2 components from 8 features capture {_fe3}% of total variance. This is typical for financial data with many correlated features."},
    {"id":4,"type":"code","marks":4,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nScale FEATURES with StandardScaler.\nFit PCA on all components.\nFind the minimum number of components to explain 80% of variance.\nPrint that number.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe4,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\npca = PCA(random_state=42)\npca.fit(X)\ncumvar = 0; nc = 0\nfor v in pca.explained_variance_ratio_:\n    cumvar += v; nc += 1\n    if cumvar >= 0.80: break\nprint(nc)\n# {_fe4}",
     "explanation":f"Cumulative variance: accumulate explained_variance_ratio_ until reaching 0.80. Need {_fe4} components to explain 80% of variance."},
    {"id":5,"type":"code","marks":4,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nPipeline:\n1. Scale FEATURES with StandardScaler\n2. Reduce to 2 PCA components (random_state=42)\n3. Apply KMeans n_clusters=3, random_state=42, n_init=10\n\nPrint cluster size distribution as a dict.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe5,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X_pca)\nprint(df2['cluster'].value_counts().to_dict())\n# {_fe5}",
     "explanation":f"Full pipeline: Scale → PCA(2) → KMeans(3). Cluster sizes: {_fe5}."},
    {"id":6,"type":"code","marks":4,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nUsing the same pipeline (StandardScaler → PCA 2 components → KMeans 3 clusters, all random_state=42, n_init=10):\n\nCompute the silhouette score on the PCA-reduced data.\nPrint rounded to 2 decimal places.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe6,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\nlabels = km.fit_predict(X_pca)\nprint(round(silhouette_score(X_pca, labels), 2))\n# {_fe6}",
     "explanation":f"Silhouette score on PCA space = {_fe6}. Higher than raw feature space because PCA removes noise dimensions."},
    {"id":7,"type":"code","marks":4,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nUsing pipeline: StandardScaler → PCA(2, random_state=42) → KMeans(3, random_state=42, n_init=10):\n\nProfile clusters by printing mean annual_income per cluster as a dict (rounded to 0 decimal places).",
     "preload":FINAL_DF_SETUP,
     "exp":_fe7,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X_pca)\nprint(df2.groupby('cluster')['annual_income'].mean().round(0).to_dict())\n# {_fe7}",
     "explanation":f"Cluster profiling: mean income per segment reveals economic segments. {_fe7}"},
    {"id":8,"type":"code","marks":5,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nUsing pipeline: StandardScaler → PCA(2, random_state=42) → KMeans(3, random_state=42, n_init=10):\n\nPrint mean is_defaulter rate per cluster as a dict (rounded to 2 decimal places).\nThis helps identify which cluster is highest risk.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe8,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X_pca)\nprint(df2.groupby('cluster')['is_defaulter'].mean().round(2).to_dict())\n# {_fe8}",
     "explanation":f"Default rates per segment: {_fe8}. This is the business value of clustering — identify high-risk customer groups."},
    {"id":9,"type":"code","marks":4,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nScale FEATURES with StandardScaler.\nApply PCA with n_components=2, random_state=42.\nFind which original feature contributes most to PC1 (highest absolute loading).\nPrint only the feature name.",
     "preload":FINAL_DF_SETUP,
     "exp":_fe9,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\npca = PCA(n_components=2, random_state=42)\npca.fit(X)\nidx = pca.components_[0].argmax()\nprint(FEATURES[idx])\n# {_fe9}",
     "explanation":f"pca.components_[0] has one loading per original feature. argmax() gives index of highest. Feature with most influence on PC1 = '{_fe9}'."},
    {"id":10,"type":"code","marks":5,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nScale FEATURES, reduce to 2 PCA components (random_state=42).\nFind the best K for KMeans (K from 2 to 6) using silhouette score.\nPrint: best_k and best_silhouette_score space-separated (score rounded to 2 decimal places).",
     "preload":FINAL_DF_SETUP,
     "exp":_fe10,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nbest_k, best_s = 2, 0\nfor k in range(2, 7):\n    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n    labels = km.fit_predict(X_pca)\n    s = silhouette_score(X_pca, labels)\n    if s > best_s: best_s = s; best_k = k\nprint(best_k, round(best_s, 2))\n# {_fe10}",
     "explanation":f"Optimal K selection via silhouette maximisation. Best K = {_fe10.split()[0]} with score = {_fe10.split()[1]}."},
    {"id":11,"type":"code","marks":5,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nUsing pipeline: StandardScaler → PCA(2, random_state=42) → KMeans(3, random_state=42, n_init=10):\n\nPrint mean credit_score per cluster as a dict (rounded to 0 decimal places).",
     "preload":FINAL_DF_SETUP,
     "exp":_fe11,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X_pca)\nprint(df2.groupby('cluster')['credit_score'].mean().round(0).to_dict())\n# {_fe11}",
     "explanation":f"Credit score profile per cluster: {_fe11}. Clusters with lower credit scores are higher-risk segments."},
    {"id":12,"type":"code","marks":5,
     "text":"df is preloaded (500 rows). FEATURES list is preloaded.\n\nUsing pipeline: StandardScaler → PCA(2, random_state=42) → KMeans(3, random_state=42, n_init=10):\n\nFind the cluster with the highest average credit_utilization.\nPrint only the cluster number (integer).",
     "preload":FINAL_DF_SETUP,
     "exp":_fe12,
     "solution":f"X = StandardScaler().fit_transform(df[FEATURES])\nX_pca = PCA(n_components=2, random_state=42).fit_transform(X)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10)\ndf2 = df.copy()\ndf2['cluster'] = km.fit_predict(X_pca)\nprint(df2.groupby('cluster')['credit_utilization'].mean().idxmax())\n# {_fe12}",
     "explanation":f"Cluster {_fe12} has highest credit utilization — likely the most financially stressed segment."},
]

# ── MAPS ─────────────────────────────────────────────────────────
M4_PRACTICE_TESTS = {1: M4_PRACTICE_1, 2: M4_PRACTICE_2, 3: M4_PRACTICE_3}
M4_FINAL_TEST     = {1: M4_FINAL}
M4_PRACTICE_DATASET = _df_p.copy()
M4_FINAL_DATASET    = _df_f.copy()
