#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss

TOPOLOGY_PATH=Path("data/features/liq_topology_v2_ml_features.parquet")
OUT=Path("data/reports/position_guardian_v4")
MODEL_OUT=Path("data/models/position_guardian_v4")
HORIZONS=[15,30,60]; SIDES=["LONG","SHORT"]
CAT_BASE=["symbol","timeframe","nearest_side"]
STATIC=[
"symbol","timeframe","nearest_side","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend_utc","current_price",
"has_upper_level","has_lower_level","has_topology","nearest_side_code","upper_distance_pct","lower_distance_pct",
"distance_advantage","signed_distance_edge","log1p_upper_distance_pct","log1p_lower_distance_pct","log1p_distance_advantage",
"upper_pool_volume","lower_pool_volume","nearest_pool_volume","farther_pool_volume","upper_total_volume","lower_total_volume",
"upper_active_levels","lower_active_levels","log1p_upper_pool_volume","log1p_lower_pool_volume","log1p_nearest_pool_volume",
"log1p_farther_pool_volume","log1p_upper_total_volume","log1p_lower_total_volume","log1p_upper_active_levels",
"log1p_lower_active_levels","pool_volume_ratio","log1p_pool_volume_ratio","distance_pressure_ratio",
"log1p_distance_pressure_ratio","topology_imbalance","total_volume_imbalance_check","active_level_difference","active_level_total"
]
DYN_SOURCES=["current_price","upper_distance_pct","lower_distance_pct","signed_distance_edge","topology_imbalance",
"upper_pool_volume","lower_pool_volume","upper_total_volume","lower_total_volume","upper_active_levels","lower_active_levels"]
FORBIDDEN=("future","forward","target","label","outcome","sweep_code","first_hit","post_hit","strong_contrarian",
"mfe","mae","endpoint","return_after","direction_1h","direction_4h")

def args():
 p=argparse.ArgumentParser()
 p.add_argument("--matrix-path",type=Path,required=True)
 p.add_argument("--timeframe",default="1h"); p.add_argument("--sample-every-minutes",type=int,default=15)
 p.add_argument("--iterations",type=int,default=500); p.add_argument("--max-train-rows",type=int,default=500000)
 p.add_argument("--validation-fraction",type=float,default=.15); p.add_argument("--test-fraction",type=float,default=.15)
 p.add_argument("--embargo-hours",type=float,default=4); p.add_argument("--exit-stop-bps",type=float,default=15)
 p.add_argument("--recovery-bps",type=float,default=12); p.add_argument("--endpoint-exit-bps",type=float,default=-8)
 p.add_argument("--min-precision",type=float,default=.45); p.add_argument("--min-coverage",type=float,default=.05)
 p.add_argument("--min-alerts",type=int,default=100); p.add_argument("--matrix-tolerance-minutes",type=int,default=20)
 return p.parse_args()

def schema_names(path): return pq.read_schema(path).names

def load_topology(a):
 schema=set(schema_names(TOPOLOGY_PATH)); need=["id","logged_at"]+STATIC
 miss=[c for c in need if c not in schema]
 if miss: raise RuntimeError(f"Topology missing: {miss}")
 df=pd.read_parquet(TOPOLOGY_PATH,columns=list(dict.fromkeys(need)),filters=[("timeframe","==",a.timeframe)])
 df["logged_at"]=pd.to_datetime(df.logged_at,utc=True).astype("datetime64[ns, UTC]")
 df["symbol"]=df.symbol.astype("string")
 df=df.sort_values(["symbol","logged_at","id"]).drop_duplicates("id").reset_index(drop=True)
 if a.sample_every_minutes>0:
  df["_bucket"]=df.logged_at.dt.floor(f"{a.sample_every_minutes}min")
  df=df.drop_duplicates(["symbol","_bucket"]).drop(columns="_bucket").reset_index(drop=True)
 return df

def add_dynamic(df):
 pieces=[]
 for _,g in df.groupby("symbol",sort=False):
  g=g.sort_values("logged_at").copy()
  for minutes,periods in [(5,1),(15,1),(30,2),(60,4)]:
   for c in DYN_SOURCES:
    if c not in g: continue
    x=pd.to_numeric(g[c],errors="coerce")
    if c=="current_price": g[f"price_return_{minutes}m_bps"]=(x/x.shift(periods)-1)*10000
    else:
     g[f"{c}_change_{minutes}m"]=x-x.shift(periods)
     if "volume" in c: g[f"{c}_growth_{minutes}m"]=x/x.shift(periods).replace(0,np.nan)-1
  p=pd.to_numeric(g.current_price,errors="coerce"); ret=p.pct_change()
  g["realized_volatility_60m_bps"]=ret.rolling(4,min_periods=2).std()*10000
  g["local_high_distance_60m_bps"]=(p/p.rolling(4,min_periods=2).max()-1)*10000
  g["local_low_distance_60m_bps"]=(p/p.rolling(4,min_periods=2).min()-1)*10000
  side=g.nearest_side.astype("string").fillna("<NA>")
  g["nearest_side_flip"]=(side!=side.shift(1)).astype("int8")
  g["nearest_side_flip_count_60m"]=g.nearest_side_flip.rolling(4,min_periods=1).sum()
  pieces.append(g)
 return pd.concat(pieces,ignore_index=True)

def load_matrix(path,a):
 if not path.exists(): raise FileNotFoundError(path)
 cols=schema_names(path); keys=[c for c in ["id","logged_at","symbol","timeframe"] if c in cols]
 if "id" not in keys and not {"logged_at","symbol"}.issubset(keys): raise RuntimeError("Matrix parquet needs id OR logged_at+symbol")
 safe=[]
 for c in cols:
  if c in keys or any(x in c.lower() for x in FORBIDDEN): continue
  safe.append(c)
 m=pd.read_parquet(path,columns=keys+safe)
 if "timeframe" in m: m=m[m.timeframe.astype(str)==a.timeframe].copy()
 if "logged_at" in m: m["logged_at"]=pd.to_datetime(m.logged_at,utc=True).astype("datetime64[ns, UTC]")
 if "symbol" in m: m["symbol"]=m.symbol.astype("string")
 matrix=[]
 for c in safe:
  if pd.api.types.is_bool_dtype(m[c]): m[c]=m[c].astype("int8"); matrix.append(c)
  elif pd.api.types.is_numeric_dtype(m[c]): matrix.append(c)
  else:
   converted=pd.to_numeric(m[c],errors="coerce")
   if converted.notna().mean()>.90: m[c]=converted; matrix.append(c)
 return m[keys+matrix].copy(),matrix

def join_matrix(topo,m,matrix_cols,a):
 if "id" in m.columns:
  out=topo.merge(m[["id"]+matrix_cols].drop_duplicates("id"),on="id",how="left",validate="one_to_one")
 else:
  left=topo.sort_values(["logged_at","symbol"]).copy(); right=m.sort_values(["logged_at","symbol"]).copy()
  out=pd.merge_asof(left,right,on="logged_at",by="symbol",direction="backward",
   tolerance=pd.Timedelta(minutes=a.matrix_tolerance_minutes),allow_exact_matches=True)
  out=out.sort_values(["symbol","logged_at","id"]).reset_index(drop=True)
 out["matrix_available"]=out[matrix_cols].notna().any(axis=1).astype("int8") if matrix_cols else 0
 return out

def add_outcomes(df,horizon,side,a):
 pieces=[]; hns=int(pd.Timedelta(minutes=horizon).value); sign=1 if side=="LONG" else -1
 for _,g in df.groupby("symbol",sort=False):
  g=g.sort_values("logged_at").copy(); t=g.logged_at.astype("int64").to_numpy(); p=g.current_price.to_numpy(float)
  y=np.full(len(g),np.nan); mfe=np.full(len(g),np.nan); mae=np.full(len(g),np.nan); end=np.full(len(g),np.nan)
  for i in range(len(g)):
   j=np.searchsorted(t,t[i]+hns,side="right")
   if j<=i+1: continue
   path=sign*(p[i+1:j]/p[i]-1)*10000
   mfe[i]=np.nanmax(path); mae[i]=np.nanmin(path); end[i]=path[-1]
   y[i]=int((mae[i]<=-a.exit_stop_bps and mfe[i]<a.recovery_bps) or end[i]<=a.endpoint_exit_bps)
  g["exit_risk"]=y; g["mfe_bps"]=mfe; g["mae_bps"]=mae; g["endpoint_bps"]=end; pieces.append(g)
 out=pd.concat(pieces,ignore_index=True).dropna(subset=["exit_risk"]); out["exit_risk"]=out.exit_risk.astype("int8")
 return out

def split(df,a):
 ts=df.logged_at; test=ts.quantile(1-a.test_fraction)
 val=ts[ts<test].quantile(1-a.validation_fraction/(1-a.test_fraction)); e=pd.Timedelta(hours=a.embargo_hours)
 tr=df[df.logged_at<=val-e].copy(); va=df[(df.logged_at>=val+e)&(df.logged_at<=test-e)].copy(); te=df[df.logged_at>=test+e].copy()
 if len(tr)>a.max_train_rows: tr=tr.iloc[np.linspace(0,len(tr)-1,a.max_train_rows,dtype=int)]
 return tr,va,te,{"train_end":str(val-e),"validation_start":str(val+e),"test_start":str(test+e),"test_end":str(ts.max())}

def prep(df,features):
 x=df[features].copy(); cats=[]
 for c in features:
  if c in CAT_BASE: x[c]=x[c].astype("string").fillna("<MISSING>").astype(str); cats.append(c)
  else: x[c]=pd.to_numeric(x[c],errors="coerce")
 return x,cats

def fit(tr,va,features,a):
 x,c=prep(tr,features); xv,_=prep(va,features)
 m=CatBoostClassifier(iterations=a.iterations,depth=8,learning_rate=.06,loss_function="Logloss",eval_metric="AUC",
  random_seed=42,verbose=False,auto_class_weights="Balanced",allow_writing_files=False)
 m.fit(Pool(x,label=tr.exit_risk,cat_features=c),eval_set=Pool(xv,label=va.exit_risk,cat_features=c),early_stopping_rounds=60,verbose=False)
 return m

def probs(m,df,features):
 x,c=prep(df,features); return m.predict_proba(Pool(x,cat_features=c))[:,1]

def choose_policy(df,p,a):
 rows=[]
 for th in np.arange(.40,.91,.025):
  pred=p>=th; n=int(pred.sum()); cov=n/len(df); prec=float(precision_score(df.exit_risk,pred,zero_division=0)); rec=float(recall_score(df.exit_risk,pred,zero_division=0))
  eligible=n>=a.min_alerts and cov>=a.min_coverage and prec>=a.min_precision
  rows.append({"threshold":round(float(th),3),"alerts":n,"coverage":cov,"precision":prec,"recall":rec,
   "lift_vs_base":prec/max(float(df.exit_risk.mean()),1e-9),"eligible":eligible,"score":2*prec+rec+min(cov,.25) if eligible else -1})
 good=[r for r in rows if r["eligible"]]
 if good: chosen=max(good,key=lambda r:(r["score"],r["precision"]))
 else:
  pool=[r for r in rows if r["alerts"]>=max(25,a.min_alerts//4)]
  chosen=dict(max(pool,key=lambda r:(r["precision"],r["recall"])) if pool else max(rows,key=lambda r:r["alerts"])); chosen["fallback"]=True
 return chosen,rows

def metrics(df,p,policy):
 pred=p>=policy["threshold"]; base=float(df.exit_risk.mean()); prec=float(precision_score(df.exit_risk,pred,zero_division=0))
 return {"rows":len(df),"base_rate":base,"alerts":int(pred.sum()),"coverage":float(pred.mean()),"precision":prec,
  "recall":float(recall_score(df.exit_risk,pred,zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(df.exit_risk,pred)),
  "lift_vs_base":prec/base if base else None,"roc_auc":float(roc_auc_score(df.exit_risk,p)),
  "pr_auc":float(average_precision_score(df.exit_risk,p)),"brier":float(brier_score_loss(df.exit_risk,p))}

def main():
 a=args(); OUT.mkdir(parents=True,exist_ok=True); MODEL_OUT.mkdir(parents=True,exist_ok=True)
 topo=add_dynamic(load_topology(a)); matrix,mcols=load_matrix(a.matrix_path,a); data=join_matrix(topo,matrix,mcols,a)
 dyn=[c for c in data.columns if any(x in c for x in ["_change_","_growth_","price_return_","realized_volatility_","local_high_","local_low_","nearest_side_flip"])]
 groups={
  "static_topology":STATIC,
  "dynamic_topology":list(dict.fromkeys(STATIC+dyn)),
  "matrix_only":list(dict.fromkeys(CAT_BASE+mcols+["matrix_available"])),
  "static_plus_matrix":list(dict.fromkeys(STATIC+mcols+["matrix_available"])),
  "dynamic_plus_matrix":list(dict.fromkeys(STATIC+dyn+mcols+["matrix_available"]))
 }
 reports=[]
 for side in SIDES:
  for h in HORIZONS:
   print(f"{side} {h}m"); ds=add_outcomes(data,h,side,a); tr,va,te,cuts=split(ds,a); ab=[]
   for name,features in groups.items():
    features=[c for c in features if c in ds.columns]
    model=fit(tr,va,features,a); pv=probs(model,va,features); pt=probs(model,te,features)
    policy,grid=choose_policy(va,pv,a); tm=metrics(te,pt,policy)
    key=f"{side.lower()}_{h}m_{name}"; d=MODEL_OUT/key; d.mkdir(parents=True,exist_ok=True); model.save_model(str(d/"model.cbm"))
    ab.append({"feature_family":name,"feature_count":len(features),"selected_policy":policy,"test_metrics":tm})
   best=max(ab,key=lambda r:(r["test_metrics"]["pr_auc"],r["test_metrics"]["roc_auc"]))
   report={"side":side,"horizon_minutes":h,"split":cuts,"matrix_path":str(a.matrix_path),"matrix_feature_count":len(mcols),
    "matrix_features":mcols,"dynamic_feature_count":len(dyn),"ablations":ab,"best_by_test_pr_auc_research_only":best["feature_family"]}
   (OUT/f"{side.lower()}_{h}m_report.json").write_text(json.dumps(report,indent=2,default=str)); reports.append(report)
 summary={"status":"research_only","target":"EXIT_RISK","design":"dynamic topology + matrix ablation","matrix_path":str(a.matrix_path),"matrix_features":mcols,"reports":reports}
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str)); print("Done:",OUT/"summary.json")

if __name__=="__main__": main()
