"""Can a length-only classifier tell synthetic rows from real ones in the pooled setting? (review point: mechanism)
Pool the 159,571 real rows and the 1,923 synthetic rows, split at random with the P1 recipe (85/7.5/7.5 stratified on any label, seeds 42/123/456/789/1011),
and within each class's positive rows train a synthetic-vs-real detector on the pooled train rows and score it on the pooled test rows (AUC).
Detectors: length only (log words, log chars) and a TF-IDF word+char n-gram logistic regression (upper bound on style cues)."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import sys
R=(sys.argv[1] if len(sys.argv)>1 else '.').rstrip('/')+'/'   # repo root
L=['severe_toxic','threat','insult','identity_hate']; ALL=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
real=pd.read_csv(R+'data/Original.csv'); syn=pd.read_csv(R+'data/Augmented.csv')
real['syn']=0; syn['syn']=1; df=pd.concat([real,syn],ignore_index=True); df['comment_text']=df.comment_text.astype(str)
assert len(real)==159571 and len(syn)==1923
df['w']=df.comment_text.str.split().str.len(); df['c']=df.comment_text.str.len()
anyl=(df[ALL].sum(1)>0).astype(int).values
def lenfeat(d): return np.c_[np.log1p(d.w.values),np.log1p(d.c.values)]
out={}
for seed in (42,123,456,789,1011):
    idx=np.arange(len(df)); tr,tmp=train_test_split(idx,test_size=0.15,random_state=seed,stratify=anyl); va,te=train_test_split(tmp,test_size=0.5,random_state=seed,stratify=anyl[tmp])
    for cls in L:
        a=tr[df.loc[tr,cls].values==1]; b=te[df.loc[te,cls].values==1]
        ya,yb=df.loc[a,'syn'].values,df.loc[b,'syn'].values
        if yb.sum()<5 or (1-yb).sum()<5: continue
        m=LogisticRegression(max_iter=1000).fit(lenfeat(df.loc[a]),ya); auc_len=roc_auc_score(yb,m.predict_proba(lenfeat(df.loc[b]))[:,1])
        v1=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True); v2=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=3,sublinear_tf=True,max_features=100000)
        Xa=hstack([v1.fit_transform(df.loc[a,'comment_text']),v2.fit_transform(df.loc[a,'comment_text'])]).tocsr(); Xb=hstack([v1.transform(df.loc[b,'comment_text']),v2.transform(df.loc[b,'comment_text'])]).tocsr()
        m2=LogisticRegression(max_iter=2000,C=3).fit(Xa,ya); auc_txt=roc_auc_score(yb,m2.predict_proba(Xb)[:,1])
        out.setdefault(cls,[]).append(dict(seed=seed,n_test=int(len(b)),n_test_syn=int(yb.sum()),auc_length=float(auc_len),auc_text=float(auc_txt)))
for cls,v in out.items():
    print(cls,'test rows %d-%d, synthetic %d-%d'%(min(x['n_test'] for x in v),max(x['n_test'] for x in v),min(x['n_test_syn'] for x in v),max(x['n_test_syn'] for x in v)),
          'AUC length-only %.3f (%.3f-%.3f)  text %.3f (%.3f-%.3f)'%(np.mean([x['auc_length'] for x in v]),min(x['auc_length'] for x in v),max(x['auc_length'] for x in v),np.mean([x['auc_text'] for x in v]),min(x['auc_text'] for x in v),max(x['auc_text'] for x in v)))
json.dump(out,open(R+'analysis/cue_check.json','w'),indent=1)
