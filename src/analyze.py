"""Reproduce descriptive tables and paired tests without models or API calls."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'data/article_scores.csv')
assert not d.duplicated(['topic','version','condition','id']).any()
assert d[['orig_bias_score','summary_bias_score','bertscore']].notna().all().all()
def mask(x,k):
 return ((x.bias_text!='center') & (x.orig_bias_score>=.15)) if k=='Suppression' else ((x.bias_text=='center') & (x.orig_bias_score<.15))
def hit(x,k):return (x.summary_bias_score<.15) if k=='Suppression' else (x.summary_bias_score>=.15)
def holm(p):
 p=np.array(p);idx=np.argsort(p);v=np.maximum.accumulate(np.minimum(1,p[idx]*np.arange(len(p),0,-1)));r=np.empty(len(p));r[idx]=v;return r
rows=[]
for (t,v,c),x in d.groupby(['topic','version','condition'],sort=False):
 s=hit(x.loc[mask(x,'Suppression')],'Suppression').mean()*100;i=hit(x.loc[mask(x,'Injection')],'Injection').mean()*100
 rows.append(dict(topic=t,version=v,condition=c,Suppression=s,Injection=i,AvgDrift=(s+i)/2,BERTScore=x.bertscore.mean(),N_supp=int(mask(x,'Suppression').sum()),N_inj=int(mask(x,'Injection').sum())))
pd.DataFrame(rows).to_csv(R/'results/descriptive.csv',index=False)
e=d[d.topic=='environment']; frames={(v,c):x.set_index('id').sort_index() for (v,c),x in e.groupby(['version','condition'])}
base=frames['revised','s1'];assert mask(base,'Suppression').sum()==61 and mask(base,'Injection').sum()==40
pairs=[('s2','s2_r'),('s3a','s3a_r'),('s3b','s3b_r'),('s3c','s3c_r'),('s4','s4_r'),('s5_cot','s5_r'),('s5_cot','s5_adv'),('s6','s6_r')]
sets={'base_vs_all':[(v,c,base,x) for (v,c),x in frames.items() if c!='s1'], 'revision_effect': [('revised',b,frames['original',a],frames['revised',b]) for a,b in pairs]}
for name,comparisons in sets.items():
 rows=[]
 for metric in ['Suppression','Injection','BERTScore']:
  sub=[]
  for version,c,a,b in comparisons:
   assert a.index.equals(b.index) and a.bias_text.equals(b.bias_text)
   assert np.allclose(a.orig_bias_score,b.orig_bias_score)
   if metric=='BERTScore':
    aa=a.bertscore.to_numpy(dtype=float);bb=b.bertscore.to_numpy(dtype=float);p=wilcoxon(aa,bb,alternative='two-sided').pvalue;scale=1
   else:
    m=mask(a,metric);aa=hit(a.loc[m],metric).to_numpy(dtype=int);bb=hit(b.loc[m],metric).to_numpy(dtype=int);z=bb-aa;n=int((z!=0).sum());p=binomtest(int((z==1).sum()),n,.5).pvalue if n else 1.;scale=100
   sub.append(dict(version=version,condition=c,outcome=metric,N=len(aa),reference_mean=aa.mean()*scale,condition_mean=bb.mean()*scale,change=(bb.mean()-aa.mean())*scale,p_raw=p))
  for row,p in zip(sub,holm([r['p_raw'] for r in sub])):row['p_holm']=p
  rows+=sub
 pd.DataFrame(rows).to_csv(R/f'results/{name}.csv',index=False)
j=pd.read_csv(R/'data/llm_judgements.csv');j.groupby(['evaluator','version','condition'])[['bias_shift','center_drift']].mean().to_csv(R/'results/llm_judgements.csv')
print('Reproduced descriptive results, 45 baseline tests, 24 revision tests, and LLM judge means.')
