import hashlib, re, io, sys
import numpy as np
import pandas as pd

def _norm(s): return re.sub(r'\s+', '', str(s).strip().lower())
def h(s):     return hashlib.sha256(_norm(s).encode()).hexdigest()
def _exec(code, globs):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    exec(code, globs); sys.stdout = old
    return buf.getvalue().strip()

# ════════════════════════════════════════════════════════════════
# BEGINNER TESTS
# ════════════════════════════════════════════════════════════════

BEGINNER_T1 = [
    {"id":1,"type":"mcq","marks":1,
     "text":"What is the output of:\n\nlen({'a': 1, 'b': 2, 'c': 3})",
     "opts":["2","3","4","6"],"ah":h("3")},
    {"id":2,"type":"mcq","marks":1,
     "text":"Which of the following data types is mutable?",
     "opts":["tuple","list","string","int"],"ah":h("list")},
    {"id":3,"type":"fill","marks":2,
     "text":"What is the output of:\n\n[1, 2, 3, 4, 5][1:4]\n\nType exactly as Python would print it.",
     "ah":h("[2, 3, 4]")},
    {"id":4,"type":"fill","marks":2,
     "text":"What is the output of:\n\nlist(range(2, 10, 3))\n\nType exactly as Python would print it.",
     "ah":h("[2, 5, 8]")},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the output of:\n\n[i**2 for i in range(5) if i % 2 != 0]",
     "ah":h("[1, 9]")},
    {"id":6,"type":"fill","marks":3,
     "text":"What does this print?\n\nd = {}\nfor i in range(5):\n    d[i] = i % 3\nprint(d)",
     "ah":h("{0: 0, 1: 1, 2: 2, 3: 0, 4: 1}")},
    {"id":7,"type":"code","marks":4,
     "text":"Preloaded:  nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n\nWrite a one-line list comprehension to extract all even numbers.\nStore in variable 'result' and print it.",
     "preload":"nums = [1,2,3,4,5,6,7,8,9,10]\n","exp":"[2, 4, 6, 8, 10]"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:  words = ['python', 'data', 'science', 'machine', 'learning']\n\nBuild a dictionary  {word: len(word)}  and print it.",
     "preload":"words = ['python','data','science','machine','learning']\n",
     "exp":"{'python': 6, 'data': 4, 'science': 7, 'machine': 7, 'learning': 8}"},
    {"id":9,"type":"code","marks":5,
     "text":"Preloaded:  nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n\nFind the most frequent element. If tie, print the smallest.\nPrint only the number.",
     "preload":"nums = [3,1,4,1,5,9,2,6,5,3,5]\n","exp":"5"},
]

BEGINNER_T2 = [
    {"id":1,"type":"mcq","marks":1,
     "text":"What is the output of:\n\n'hello'.upper()",
     "opts":["HELLO","hello","Hello","hELLO"],"ah":h("HELLO")},
    {"id":2,"type":"mcq","marks":1,
     "text":"What does len('python') return?",
     "opts":["5","6","7","8"],"ah":h("6")},
    {"id":3,"type":"fill","marks":2,
     "text":"What is the output of:\n\n'hello world'.split()\n\nType exactly as Python would print it.",
     "ah":h("['hello', 'world']")},
    {"id":4,"type":"fill","marks":2,
     "text":"What is the output of:\n\n'-'.join(['a', 'b', 'c'])",
     "ah":h("a-b-c")},
    {"id":5,"type":"fill","marks":3,
     "text":"What does this print?\n\ndef f(x):\n    return x * 2\nprint(f(5))",
     "ah":h("10")},
    {"id":6,"type":"fill","marks":3,
     "text":"What is the output of:\n\n'abcde'[::-1]",
     "ah":h("edcba")},
    {"id":7,"type":"code","marks":4,
     "text":"Preloaded:  text = 'Hello World'\n\nCount the number of vowels (a,e,i,o,u) in text.\nPrint only the count.",
     "preload":"text = 'Hello World'\n","exp":"3"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:  sentence = 'Python is awesome'\n\nReverse the order of words in the sentence.\nPrint the result.",
     "preload":"sentence = 'Python is awesome'\n","exp":"awesome is Python"},
    {"id":9,"type":"code","marks":5,
     "text":"Preloaded:  word = 'racecar'\n\nCheck if word is a palindrome.\nPrint True or False.",
     "preload":"word = 'racecar'\n","exp":"True"},
]

BEGINNER_T3 = [
    {"id":1,"type":"mcq","marks":1,
     "text":"Which of the following is immutable?",
     "opts":["list","dict","tuple","set"],"ah":h("tuple")},
    {"id":2,"type":"mcq","marks":1,
     "text":"What is the output of:\n\n{1, 2, 3} & {2, 3, 4}",
     "opts":["{1}","{2, 3}","{1, 2, 3, 4}","{4}"],"ah":h("{2, 3}")},
    {"id":3,"type":"fill","marks":2,
     "text":"What is the output of:\n\nsorted([3, 1, 4, 1, 5], reverse=True)\n\nType exactly as Python would print it.",
     "ah":h("[5, 4, 3, 1, 1]")},
    {"id":4,"type":"fill","marks":2,
     "text":"What is the output of:\n\nsorted(set([1, 2, 2, 3, 3, 3]))\n\nType exactly as Python would print it.",
     "ah":h("[1, 2, 3]")},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the output of:\n\n(1, 2, 3) + (4, 5)\n\nType exactly as Python would print it.",
     "ah":h("(1, 2, 3, 4, 5)")},
    {"id":6,"type":"fill","marks":3,
     "text":"What is the output of:\n\nmax([3, 1, 4, 1, 5, 9, 2, 6])",
     "ah":h("9")},
    {"id":7,"type":"code","marks":4,
     "text":"Preloaded:  nums = [1, 2, 2, 3, 3, 3, 4]\n\nRemove duplicates while preserving the original order.\nStore in variable 'result' and print it.",
     "preload":"nums = [1,2,2,3,3,3,4]\n","exp":"[1, 2, 3, 4]"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:\n  a = [1, 2, 3, 4]\n  b = [3, 4, 5, 6]\n\nFind common elements between a and b.\nPrint as a sorted list.",
     "preload":"a=[1,2,3,4]\nb=[3,4,5,6]\n","exp":"[3, 4]"},
    {"id":9,"type":"code","marks":5,
     "text":"Preloaded:  nested = [[1, 2], [3, 4], [5, 6]]\n\nFlatten nested into a single list using a list comprehension.\nPrint the result.",
     "preload":"nested=[[1,2],[3,4],[5,6]]\n","exp":"[1, 2, 3, 4, 5, 6]"},
]

BEGINNER_T4 = [
    {"id":1,"type":"mcq","marks":1,
     "text":"What is the output of:\n\nbool(0)",
     "opts":["True","False","0","None"],"ah":h("False")},
    {"id":2,"type":"mcq","marks":1,
     "text":"What is the output of:\n\nbool('')",
     "opts":["True","False","''","None"],"ah":h("False")},
    {"id":3,"type":"fill","marks":2,
     "text":"What is the output of:\n\n5 if 3 > 2 else 10",
     "ah":h("5")},
    {"id":4,"type":"fill","marks":2,
     "text":"What is the output of:\n\nsum(range(1, 6))",
     "ah":h("15")},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the output of:\n\n[x for x in range(10) if x % 3 == 0]\n\nType exactly as Python would print it.",
     "ah":h("[0, 3, 6, 9]")},
    {"id":6,"type":"fill","marks":3,
     "text":"In FizzBuzz output for numbers 1 to 15,\nhow many numbers print exactly 'Fizz' (not 'FizzBuzz')?\n\nType only the number.",
     "ah":h("4")},
    {"id":7,"type":"code","marks":4,
     "text":"Write code to find all prime numbers from 2 to 20 (inclusive).\nStore in list 'result' and print it.",
     "preload":"","exp":"[2, 3, 5, 7, 11, 13, 17, 19]"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:  nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\n\nFind the second largest unique number.\nPrint only the number.",
     "preload":"nums=[3,1,4,1,5,9,2,6,5,3,5]\n","exp":"6"},
    {"id":9,"type":"code","marks":5,
     "text":"Write a function is_prime(n) that returns True if n is prime, else False.\nThen print the count of prime numbers between 1 and 50 (inclusive).",
     "preload":"","exp":"15"},
]

BEGINNER_T5 = [
    {"id":1,"type":"mcq","marks":1,
     "text":"What is the data type of:  3.14",
     "opts":["int","float","double","str"],"ah":h("float")},
    {"id":2,"type":"mcq","marks":1,
     "text":"Which function returns a NEW sorted list without modifying the original?",
     "opts":["list.sort()","sorted()","order()","arrange()"],"ah":h("sorted()")},
    {"id":3,"type":"fill","marks":2,
     "text":"What is the output of:\n\nabs(-7)",
     "ah":h("7")},
    {"id":4,"type":"fill","marks":2,
     "text":"What is the output of:\n\nround(3.567, 2)",
     "ah":h("3.57")},
    {"id":5,"type":"fill","marks":3,
     "text":"What is the output of:\n\nlist(map(lambda x: x**2, [1, 2, 3]))\n\nType exactly as Python would print it.",
     "ah":h("[1, 4, 9]")},
    {"id":6,"type":"fill","marks":3,
     "text":"What is the output of:\n\nlist(filter(lambda x: x > 3, [1, 2, 3, 4, 5]))\n\nType exactly as Python would print it.",
     "ah":h("[4, 5]")},
    {"id":7,"type":"code","marks":4,
     "text":"Preloaded:\n  keys   = ['a', 'b', 'c']\n  values = [1, 2, 3]\n\nUse zip to combine them into a dictionary and print it.",
     "preload":"keys=['a','b','c']\nvalues=[1,2,3]\n","exp":"{'a': 1, 'b': 2, 'c': 3}"},
    {"id":8,"type":"code","marks":4,
     "text":"Preloaded:  d = {'a': [1, 2], 'b': [3, 4]}\n\nFlatten all values into a single list and print it.",
     "preload":"d={'a':[1,2],'b':[3,4]}\n","exp":"[1, 2, 3, 4]"},
    {"id":9,"type":"code","marks":5,
     "text":"Preloaded:\n  students = [{'name':'Alice','score':85}, {'name':'Bob','score':72}, {'name':'Charlie','score':91}]\n\nSort by score descending.\nPrint a list of names in that order.",
     "preload":"students=[{'name':'Alice','score':85},{'name':'Bob','score':72},{'name':'Charlie','score':91}]\n",
     "exp":"['Charlie', 'Alice', 'Bob']"},
]

# ════════════════════════════════════════════════════════════════
# INTERMEDIATE TESTS
# ════════════════════════════════════════════════════════════════

# ── T1: Credit Line Assignment (seed=42) ─────────────────────────
np.random.seed(42); _n1=200
_df1=pd.DataFrame({'customer_id':range(1001,1201),'age':np.random.randint(22,65,_n1),'income':np.random.randint(300000,2000000,_n1),'bureau_score':np.random.randint(500,850,_n1),'existing_loans':np.random.randint(0,6,_n1),'employment_type':np.random.choice(['Salaried','Self-Employed','Business','Freelancer'],_n1,p=[0.5,0.2,0.2,0.1]),'city':np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'],_n1),'credit_line_assigned':np.random.randint(50000,1000000,_n1)})
_G1={"df":_df1.copy(),"pd":pd,"np":np}
_df1b=_df1.copy(); _df1b['income_band']=pd.cut(_df1b['income'],3,labels=['L','M','H'])
DF1_SETUP="import pandas as pd,numpy as np\nnp.random.seed(42)\n_n=200\ndf=pd.DataFrame({'customer_id':range(1001,1201),'age':np.random.randint(22,65,_n),'income':np.random.randint(300000,2000000,_n),'bureau_score':np.random.randint(500,850,_n),'existing_loans':np.random.randint(0,6,_n),'employment_type':np.random.choice(['Salaried','Self-Employed','Business','Freelancer'],_n,p=[0.5,0.2,0.2,0.1]),'city':np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'],_n),'credit_line_assigned':np.random.randint(50000,1000000,_n)})\n"
_i1e7=_exec("df['sb']=pd.cut(df['bureau_score'],5,labels=['1','2','3','4','5'])\nprint(df.groupby('sb')['credit_line_assigned'].mean().idxmax())",{**_G1,"df":_df1.copy()})
_i1e8=_exec("print(df.groupby('city')['income'].median().nlargest(3).index.tolist())",{**_G1,"df":_df1.copy()})
_i1e9=_exec("print(round(df['income'].corr(df['credit_line_assigned']),2))",{**_G1,"df":_df1.copy()})
_i1e10=_exec("print(len(df[(df['bureau_score']<600)&(df['existing_loans']>2)]))",{**_G1,"df":_df1.copy()})

INTERMEDIATE_T1=[
    {"id":1,"type":"mcq","marks":1,"text":"df['employment_type'].nunique()\n\nWhat does this return?","opts":["2","3","4","5"],"ah":h(str(_df1['employment_type'].nunique()))},
    {"id":2,"type":"mcq","marks":2,"text":"Which city has the highest average credit_line_assigned?\n(Use the preloaded df)","opts":["Mumbai","Delhi","Bangalore","Chennai","Hyderabad"],"ah":h(_df1.groupby('city')['credit_line_assigned'].mean().idxmax())},
    {"id":3,"type":"fill","marks":2,"text":"What is the output of:\n\ndf['bureau_score'].median()\n\nType the exact value (e.g. 675.0)","ah":h(str(_df1['bureau_score'].median()))},
    {"id":4,"type":"fill","marks":3,"text":"How many customers have exactly 0 existing loans?\n\ndf[df['existing_loans'] == 0].shape[0]\n\nType the integer.","ah":h(str(_df1[_df1['existing_loans']==0].shape[0]))},
    {"id":5,"type":"fill","marks":3,"text":"What does this return?\n\ndf.groupby('employment_type')['credit_line_assigned'].mean().idxmax()\n\nType the exact string.","ah":h(_df1.groupby('employment_type')['credit_line_assigned'].mean().idxmax())},
    {"id":6,"type":"mcq","marks":4,"text":"After running:\ndf['income_band'] = pd.cut(df['income'], 3, labels=['L','M','H'])\n\nWhich income band has the most customers?","opts":["L","M","H"],"ah":h(str(_df1b['income_band'].value_counts().idxmax()))},
    {"id":7,"type":"code","marks":4,"text":"df is preloaded (200 rows).\n\nCut bureau_score into 5 equal-width bands labeled '1' to '5'.\nFind the band with the highest average credit_line_assigned.\nPrint only the label.","preload":DF1_SETUP,"exp":_i1e7},
    {"id":8,"type":"code","marks":4,"text":"df is preloaded (200 rows).\n\nFind the top 3 cities by median income.\nPrint as a Python list.","preload":DF1_SETUP,"exp":_i1e8},
    {"id":9,"type":"code","marks":4,"text":"df is preloaded (200 rows).\n\nCompute Pearson correlation between 'income' and 'credit_line_assigned'.\nPrint rounded to 2 decimal places.","preload":DF1_SETUP,"exp":_i1e9},
    {"id":10,"type":"code","marks":5,"text":"df is preloaded (200 rows).\n\nFlag customers as High Risk if:\n    bureau_score < 600  AND  existing_loans > 2\n\nPrint the count of High Risk customers.","preload":DF1_SETUP,"exp":_i1e10},
]

# ── T2: Credit Line Advanced (seed=99) ───────────────────────────
np.random.seed(99); _n2=250
_df2=pd.DataFrame({'customer_id':range(2001,2251),'age':np.random.randint(25,65,_n2),'annual_income':np.random.randint(200000,3000000,_n2),'bureau_score':np.random.randint(550,900,_n2),'credit_utilization':np.round(np.random.uniform(0.1,0.95,_n2),2),'num_credit_cards':np.random.randint(1,8,_n2),'months_on_book':np.random.randint(6,120,_n2),'employment_type':np.random.choice(['Salaried','Self-Employed','Business','Retired'],_n2,p=[0.55,0.2,0.15,0.1]),'city_tier':np.random.choice(['Tier1','Tier2','Tier3'],_n2,p=[0.4,0.35,0.25]),'credit_line_assigned':np.random.randint(25000,800000,_n2)})
_G2={"df":_df2.copy(),"pd":pd,"np":np}
DF2_SETUP="import pandas as pd,numpy as np\nnp.random.seed(99)\n_n=250\ndf=pd.DataFrame({'customer_id':range(2001,2251),'age':np.random.randint(25,65,_n),'annual_income':np.random.randint(200000,3000000,_n),'bureau_score':np.random.randint(550,900,_n),'credit_utilization':np.round(np.random.uniform(0.1,0.95,_n),2),'num_credit_cards':np.random.randint(1,8,_n),'months_on_book':np.random.randint(6,120,_n),'employment_type':np.random.choice(['Salaried','Self-Employed','Business','Retired'],_n,p=[0.55,0.2,0.15,0.1]),'city_tier':np.random.choice(['Tier1','Tier2','Tier3'],_n,p=[0.4,0.35,0.25]),'credit_line_assigned':np.random.randint(25000,800000,_n)})\n"
_i2e7=_exec("print(df.groupby('employment_type')['credit_line_assigned'].mean().idxmax())",{**_G2,"df":_df2.copy()})
_i2e8=_exec("df['ub']=pd.cut(df['credit_utilization'],bins=[0,0.3,0.6,1.0],labels=['Low','Medium','High'])\nprint(df['ub'].value_counts().idxmax())",{**_G2,"df":_df2.copy()})
_i2e9=_exec("df['sb']=pd.cut(df['bureau_score'],5,labels=['1','2','3','4','5'])\nprint(df.groupby('sb')['credit_line_assigned'].mean().idxmax())",{**_G2,"df":_df2.copy()})
_i2e10=_exec("t1=df[df['city_tier']=='Tier1']\nprint(t1.groupby('employment_type')['credit_line_assigned'].mean().idxmax())",{**_G2,"df":_df2.copy()})

INTERMEDIATE_T2=[
    {"id":1,"type":"mcq","marks":1,"text":"df['city_tier'].nunique()\n\nWhat does this return?","opts":["2","3","4","5"],"ah":h(str(_df2['city_tier'].nunique()))},
    {"id":2,"type":"mcq","marks":2,"text":"Which city_tier has the highest average credit_line_assigned?\n(Use the preloaded df)","opts":["Tier1","Tier2","Tier3"],"ah":h(_df2.groupby('city_tier')['credit_line_assigned'].mean().idxmax())},
    {"id":3,"type":"fill","marks":2,"text":"What is the output of:\n\nround(df['credit_utilization'].mean(), 2)\n\nType the exact value.","ah":h(str(round(_df2['credit_utilization'].mean(),2)))},
    {"id":4,"type":"fill","marks":3,"text":"How many customers have 5 or more credit cards?\n\ndf[df['num_credit_cards'] >= 5].shape[0]\n\nType the integer.","ah":h(str(_df2[_df2['num_credit_cards']>=5].shape[0]))},
    {"id":5,"type":"fill","marks":3,"text":"What is the output of:\n\nround(df['annual_income'].corr(df['credit_line_assigned']), 2)\n\nType the exact value.","ah":h(str(round(_df2['annual_income'].corr(_df2['credit_line_assigned']),2)))},
    {"id":6,"type":"mcq","marks":4,"text":"Which city_tier has the most customers in df?","opts":["Tier1","Tier2","Tier3"],"ah":h(_df2['city_tier'].value_counts().idxmax())},
    {"id":7,"type":"code","marks":4,"text":"df is preloaded (250 rows).\nColumns: employment_type, credit_line_assigned (and others).\n\nFind the employment_type with the highest average credit_line_assigned.\nPrint only the name.","preload":DF2_SETUP,"exp":_i2e7},
    {"id":8,"type":"code","marks":4,"text":"df is preloaded (250 rows).\n\nCreate 'util_bucket' using pd.cut on credit_utilization:\n    bins: [0, 0.3, 0.6, 1.0]   labels: ['Low', 'Medium', 'High']\n\nPrint the bucket name with the most customers.","preload":DF2_SETUP,"exp":_i2e8},
    {"id":9,"type":"code","marks":4,"text":"df is preloaded (250 rows).\n\nCut bureau_score into 5 equal-width bands labeled '1' to '5'.\nPrint the score band with the highest average credit_line_assigned.","preload":DF2_SETUP,"exp":_i2e9},
    {"id":10,"type":"code","marks":5,"text":"df is preloaded (250 rows).\n\nFilter to Tier1 city customers only.\nAmong them, find the employment_type with the highest average credit_line_assigned.\nPrint only the name.","preload":DF2_SETUP,"exp":_i2e10},
]

# ── T3: 90+ DPD Prediction (seed=123) ────────────────────────────
np.random.seed(123); _n3=300
_df3=pd.DataFrame({'customer_id':range(3001,3301),'age':np.random.randint(22,60,_n3),'loan_amount':np.random.randint(50000,1500000,_n3),'tenure_months':np.random.choice([12,24,36,48,60],_n3),'bureau_score':np.random.randint(450,850,_n3),'existing_emi':np.random.randint(0,50000,_n3),'employment_type':np.random.choice(['Salaried','Self-Employed','Business'],_n3,p=[0.6,0.25,0.15]),'dpd_90_flag':np.random.choice([0,1],_n3,p=[0.75,0.25]),'foir':np.round(np.random.uniform(0.2,0.8,_n3),2),'loan_type':np.random.choice(['Personal','Auto','Home','Education'],_n3,p=[0.4,0.3,0.2,0.1])})
_G3={"df":_df3.copy(),"pd":pd,"np":np}
_hfoir3=_df3[_df3['foir']>0.6]['dpd_90_flag'].mean(); _overall3=_df3['dpd_90_flag'].mean()
DF3_SETUP="import pandas as pd,numpy as np\nnp.random.seed(123)\n_n=300\ndf=pd.DataFrame({'customer_id':range(3001,3301),'age':np.random.randint(22,60,_n),'loan_amount':np.random.randint(50000,1500000,_n),'tenure_months':np.random.choice([12,24,36,48,60],_n),'bureau_score':np.random.randint(450,850,_n),'existing_emi':np.random.randint(0,50000,_n),'employment_type':np.random.choice(['Salaried','Self-Employed','Business'],_n,p=[0.6,0.25,0.15]),'dpd_90_flag':np.random.choice([0,1],_n,p=[0.75,0.25]),'foir':np.round(np.random.uniform(0.2,0.8,_n),2),'loan_type':np.random.choice(['Personal','Auto','Home','Education'],_n,p=[0.4,0.3,0.2,0.1])})\n"
_i3e7=_exec("print(df.groupby('employment_type')['dpd_90_flag'].mean().idxmax())",{**_G3,"df":_df3.copy()})
_i3e8=_exec("print(round(df['bureau_score'].corr(df['dpd_90_flag']),2))",{**_G3,"df":_df3.copy()})
_i3e9=_exec("df['risk']=pd.cut(df['bureau_score'],bins=[0,550,700,1000],labels=['High','Medium','Low'])\nprint(df['risk'].value_counts().idxmax())",{**_G3,"df":_df3.copy()})
_i3e10=_exec("print(df.groupby('tenure_months')['dpd_90_flag'].mean().idxmax())",{**_G3,"df":_df3.copy()})

INTERMEDIATE_T3=[
    {"id":1,"type":"mcq","marks":1,"text":"df['dpd_90_flag'].value_counts().idxmax()\n\nWhat does this return? (0=no default, 1=defaulted)","opts":["0","1","2","None"],"ah":h(str(int(_df3['dpd_90_flag'].value_counts().idxmax())))},
    {"id":2,"type":"mcq","marks":2,"text":"Which loan_type has the highest average loan_amount?\n(Use the preloaded df)","opts":["Personal","Auto","Home","Education"],"ah":h(_df3.groupby('loan_type')['loan_amount'].mean().idxmax())},
    {"id":3,"type":"fill","marks":2,"text":"What is the overall 90+ DPD rate?\n\nround(df['dpd_90_flag'].mean(), 2)\n\nType the exact value.","ah":h(str(round(_df3['dpd_90_flag'].mean(),2)))},
    {"id":4,"type":"fill","marks":3,"text":"What is the DPD rate for customers with bureau_score < 600?\n\nround(df[df['bureau_score'] < 600]['dpd_90_flag'].mean(), 2)\n\nType the exact value.","ah":h(str(round(_df3[_df3['bureau_score']<600]['dpd_90_flag'].mean(),2)))},
    {"id":5,"type":"fill","marks":3,"text":"Which loan_type has the highest total count of defaults (sum of dpd_90_flag)?\n\ndf.groupby('loan_type')['dpd_90_flag'].sum().idxmax()\n\nType the exact string.","ah":h(_df3.groupby('loan_type')['dpd_90_flag'].sum().idxmax())},
    {"id":6,"type":"mcq","marks":4,"text":"Compare DPD rate for customers with foir > 0.6 vs the overall DPD rate.\nWhich statement is correct?","opts":["Higher than overall","Lower than overall","Same as overall"],"ah":h("Lower than overall" if _hfoir3<_overall3 else "Higher than overall")},
    {"id":7,"type":"code","marks":4,"text":"df is preloaded (300 rows).\nTarget: dpd_90_flag (1=defaulted, 0=not)\n\nFind the employment_type with the highest DPD rate.\nPrint only the name.","preload":DF3_SETUP,"exp":_i3e7},
    {"id":8,"type":"code","marks":4,"text":"df is preloaded (300 rows).\n\nCompute Pearson correlation between 'bureau_score' and 'dpd_90_flag'.\nPrint rounded to 2 decimal places.","preload":DF3_SETUP,"exp":_i3e8},
    {"id":9,"type":"code","marks":4,"text":"df is preloaded (300 rows).\n\nCreate 'risk' column using pd.cut on bureau_score:\n    bins: [0, 550, 700, 1000]   labels: ['High', 'Medium', 'Low']\n\nPrint the risk segment with the most customers.","preload":DF3_SETUP,"exp":_i3e9},
    {"id":10,"type":"code","marks":5,"text":"df is preloaded (300 rows).\n\nFind the tenure_months value with the highest average DPD rate.\nPrint only the tenure value.","preload":DF3_SETUP,"exp":_i3e10},
]

# ── T4: 90+ DPD Temporal (seed=456) ──────────────────────────────
np.random.seed(456); _n4=250
_m4=pd.date_range('2022-01',periods=24,freq='ME')
_df4=pd.DataFrame({'customer_id':range(4001,4251),'disbursement_month':np.random.choice(_m4,_n4),'bureau_score':np.random.randint(500,850,_n4),'loan_amount':np.random.randint(100000,2000000,_n4),'dpd_90_flag':np.random.choice([0,1],_n4,p=[0.72,0.28]),'state':np.random.choice(['MH','DL','KA','TN','UP'],_n4),'product':np.random.choice(['PL','AL','HL'],_n4,p=[0.5,0.3,0.2]),'vintage_months':np.random.randint(1,25,_n4)})
_df4['year']=_df4['disbursement_month'].dt.year; _df4['quarter']=_df4['disbursement_month'].dt.quarter
_G4={"df":_df4.copy(),"pd":pd,"np":np}
DF4_SETUP="import pandas as pd,numpy as np\nnp.random.seed(456)\n_n=250\n_m=pd.date_range('2022-01',periods=24,freq='ME')\ndf=pd.DataFrame({'customer_id':range(4001,4251),'disbursement_month':np.random.choice(_m,_n),'bureau_score':np.random.randint(500,850,_n),'loan_amount':np.random.randint(100000,2000000,_n),'dpd_90_flag':np.random.choice([0,1],_n,p=[0.72,0.28]),'state':np.random.choice(['MH','DL','KA','TN','UP'],_n),'product':np.random.choice(['PL','AL','HL'],_n,p=[0.5,0.3,0.2]),'vintage_months':np.random.randint(1,25,_n)})\ndf['year']=df['disbursement_month'].dt.year\ndf['quarter']=df['disbursement_month'].dt.quarter\n"
_i4e7=_exec("print(df.groupby('state')['dpd_90_flag'].mean().idxmin())",{**_G4,"df":_df4.copy()})
_i4e8=_exec("print(round(df['vintage_months'].corr(df['dpd_90_flag']),2))",{**_G4,"df":_df4.copy()})
_i4e9=_exec("df['vb']=pd.cut(df['vintage_months'],bins=[0,6,12,25],labels=['Early','Mid','Late'])\nprint(df.groupby('vb')['dpd_90_flag'].mean().idxmax())",{**_G4,"df":_df4.copy()})
_i4e10=_exec("print(df.groupby('quarter').size().idxmax())",{**_G4,"df":_df4.copy()})

INTERMEDIATE_T4=[
    {"id":1,"type":"mcq","marks":1,"text":"df['product'].nunique()\n\nWhat does this return?","opts":["2","3","4","5"],"ah":h(str(_df4['product'].nunique()))},
    {"id":2,"type":"mcq","marks":2,"text":"Which state has the highest 90+ DPD rate?\n(Use the preloaded df)","opts":["MH","DL","KA","TN","UP"],"ah":h(_df4.groupby('state')['dpd_90_flag'].mean().idxmax())},
    {"id":3,"type":"fill","marks":2,"text":"What is the overall 90+ DPD rate in this dataset?\n\nround(df['dpd_90_flag'].mean(), 2)\n\nType the exact value.","ah":h(str(round(_df4['dpd_90_flag'].mean(),2)))},
    {"id":4,"type":"fill","marks":3,"text":"What is the DPD rate for early customers (vintage_months <= 6)?\n\nround(df[df['vintage_months'] <= 6]['dpd_90_flag'].mean(), 2)\n\nType the exact value.","ah":h(str(round(_df4[_df4['vintage_months']<=6]['dpd_90_flag'].mean(),2)))},
    {"id":5,"type":"fill","marks":3,"text":"Which year had the highest average DPD rate?\n\ndf.groupby('year')['dpd_90_flag'].mean().idxmax()\n\nType the year (e.g. 2022).","ah":h(str(_df4.groupby('year')['dpd_90_flag'].mean().idxmax()))},
    {"id":6,"type":"mcq","marks":4,"text":"Which product has the lowest average loan_amount?","opts":["PL","AL","HL"],"ah":h(_df4.groupby('product')['loan_amount'].mean().idxmin())},
    {"id":7,"type":"code","marks":4,"text":"df is preloaded (250 rows).\ndf has: state, dpd_90_flag, year, quarter (and others).\n\nFind the state with the LOWEST 90+ DPD rate.\nPrint only the state code.","preload":DF4_SETUP,"exp":_i4e7},
    {"id":8,"type":"code","marks":4,"text":"df is preloaded (250 rows).\n\nCompute Pearson correlation between 'vintage_months' and 'dpd_90_flag'.\nPrint rounded to 2 decimal places.","preload":DF4_SETUP,"exp":_i4e8},
    {"id":9,"type":"code","marks":4,"text":"df is preloaded (250 rows).\n\nCreate 'vbucket' using pd.cut on vintage_months:\n    bins: [0, 6, 12, 25]   labels: ['Early', 'Mid', 'Late']\n\nPrint the vintage bucket with the highest DPD rate.","preload":DF4_SETUP,"exp":_i4e9},
    {"id":10,"type":"code","marks":5,"text":"df is preloaded (250 rows).\nColumn 'quarter' is already computed (1,2,3,4).\n\nFind the quarter with the most loan disbursements.\nPrint only the quarter number.","preload":DF4_SETUP,"exp":_i4e10},
]

# ── T5: Credit Card Spend Pattern (seed=789) ─────────────────────
np.random.seed(789); _n5=500
_df5=pd.DataFrame({'customer_id':np.random.choice(range(5001,5101),_n5),'month':np.random.choice(pd.date_range('2023-01',periods=12,freq='ME'),_n5),'category':np.random.choice(['Shopping','Food','Travel','Fuel','Entertainment','Healthcare'],_n5),'spend_amount':np.random.randint(500,50000,_n5),'transaction_count':np.random.randint(1,20,_n5),'city':np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'],_n5)})
_G5={"df":_df5.copy(),"pd":pd,"np":np}
DF5_SETUP="import pandas as pd,numpy as np\nnp.random.seed(789)\n_n=500\ndf=pd.DataFrame({'customer_id':np.random.choice(range(5001,5101),_n),'month':np.random.choice(pd.date_range('2023-01',periods=12,freq='ME'),_n),'category':np.random.choice(['Shopping','Food','Travel','Fuel','Entertainment','Healthcare'],_n),'spend_amount':np.random.randint(500,50000,_n),'transaction_count':np.random.randint(1,20,_n),'city':np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad'],_n)})\n"
_i5e7=_exec("df['ms']=df['month'].dt.strftime('%Y-%m')\nprint(df.groupby('ms')['spend_amount'].sum().idxmax())",{**_G5,"df":_df5.copy()})
_i5e8=_exec("print(round(df.groupby('customer_id')['spend_amount'].sum().mean(),2))",{**_G5,"df":_df5.copy()})
_i5e9=_exec("print(round(df['spend_amount'].corr(df['transaction_count']),2))",{**_G5,"df":_df5.copy()})
_i5e10=_exec("cats=df.groupby('customer_id')['category'].nunique()\nprint(len(cats[cats>=4]))",{**_G5,"df":_df5.copy()})

INTERMEDIATE_T5=[
    {"id":1,"type":"mcq","marks":1,"text":"df['category'].nunique()\n\nWhat does this return?","opts":["4","5","6","7"],"ah":h(str(_df5['category'].nunique()))},
    {"id":2,"type":"mcq","marks":2,"text":"Which category has the highest total spend_amount?\n(Use the preloaded df)","opts":["Shopping","Food","Travel","Fuel","Entertainment","Healthcare"],"ah":h(_df5.groupby('category')['spend_amount'].sum().idxmax())},
    {"id":3,"type":"fill","marks":2,"text":"What is the output of:\n\nround(df['spend_amount'].mean(), 0)\n\nType the exact value (e.g. 25000.0).","ah":h(str(round(_df5['spend_amount'].mean(),0)))},
    {"id":4,"type":"fill","marks":3,"text":"How many unique customers are in this dataset?\n\ndf['customer_id'].nunique()\n\nType the integer.","ah":h(str(_df5['customer_id'].nunique()))},
    {"id":5,"type":"fill","marks":3,"text":"Which city has the highest average transaction_count?\n\ndf.groupby('city')['transaction_count'].mean().idxmax()\n\nType the exact city name.","ah":h(_df5.groupby('city')['transaction_count'].mean().idxmax())},
    {"id":6,"type":"mcq","marks":4,"text":"Which category has the highest average spend_amount?","opts":["Shopping","Food","Travel","Fuel","Entertainment","Healthcare"],"ah":h(_df5.groupby('category')['spend_amount'].mean().idxmax())},
    {"id":7,"type":"code","marks":4,"text":"df is preloaded (500 rows).\nColumn 'month' is a datetime.\n\nFind the month with the highest total spend_amount.\nFormat output as: YYYY-MM  (e.g. 2023-03)\nPrint only the month string.","preload":DF5_SETUP,"exp":_i5e7},
    {"id":8,"type":"code","marks":4,"text":"df is preloaded (500 rows).\n\nCompute average total spend per customer\n(sum each customer's spend first, then take the mean).\nPrint rounded to 2 decimal places.","preload":DF5_SETUP,"exp":_i5e8},
    {"id":9,"type":"code","marks":4,"text":"df is preloaded (500 rows).\n\nCompute Pearson correlation between 'spend_amount' and 'transaction_count'.\nPrint rounded to 2 decimal places.","preload":DF5_SETUP,"exp":_i5e9},
    {"id":10,"type":"code","marks":5,"text":"df is preloaded (500 rows).\n\nFind customers who have spent in 4 or more distinct categories.\nPrint the count of such customers.","preload":DF5_SETUP,"exp":_i5e10},
]

# ── MAPS ─────────────────────────────────────────────────────────
BEGINNER_TESTS     = {1:BEGINNER_T1, 2:BEGINNER_T2, 3:BEGINNER_T3, 4:BEGINNER_T4, 5:BEGINNER_T5}
INTERMEDIATE_TESTS = {1:INTERMEDIATE_T1, 2:INTERMEDIATE_T2, 3:INTERMEDIATE_T3, 4:INTERMEDIATE_T4, 5:INTERMEDIATE_T5}
