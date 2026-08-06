#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, confusion_matrix

FEATURE_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")
EVENT_DIR = Path("data/research/topology_v2_squeeze_grid/tf_1h__future_60m__precursor_5m__q_0p975")
EVENT_DATASET_PATH = EVENT_DIR / "squeeze_event_dataset.parquet"
DETECTED_EVENTS_PATH = EVENT_DIR / "detected_squeeze_events.parquet"
OUT = Path("data/reports/topology_v2_causal_research")

FORBIDDEN = (
    "sweep_code_", "first_hit_seconds_", "post_hit_continuation_",
    "strong_contrarian_", "direction_", "forward_return_", "target_"
)
CAT_CANDIDATES = ["symbol", "timeframe", "nearest_side"]
CALENDAR = ["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend_utc"]
DISTANCE = [
    "upper_distance_pct","lower_distance_pct","distance_advantage",
    "signed_distance_edge","log1p_upper_distance_pct",
    "log1p_lower_distance_pct","log1p_distance_advantage",
]
TOPOLOGY = [
    "has_upper_level","has_lower_level","has_topology","nearest_side_code",
    *DISTANCE,
    "upper_pool_volume","lower_pool_volume","nearest_pool_volume",
    "farther_pool_volume","upper_total_volume","lower_total_volume",
    "upper_active_levels","lower_active_levels",
    "log1p_upper_pool_volume","log1p_lower_pool_volume",
    "log1p_nearest_pool_volume","log1p_farther_pool_volume",
    "log1p_upper_total_volume","log1p_lower_total_volume",
    "log1p_upper_active_levels","log1p_lower_active_levels",
    "pool_volume_ratio","log1p_pool_volume_ratio",
    "distance_pressure_ratio","log1p_distance_pressure_ratio",
    "topology_imbalance","total_volume_imbalance_check",
    "active_level_difference","active_level_total",
]
PRICE_COLS = [
    "id","logged_at","symbol","timeframe","current_price","nearest_side",
    "nearest_upper_price","nearest_lower_price"
]
CLASSES = [-1,0,1]
RNG = np.random.default_rng(42)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--max-train-rows",type=int,default=600_000)
    p.add_argument("--iterations",type=int,default=450)
    p.add_argument("--outer-fraction",type=float,default=.15)
    p.add_argument("--validation-fraction",type=float,default=.15)
    p.add_argument("--embargo-hours",type=float,default=4)
    p.add_argument("--event-window-minutes",type=int,default=60)
    p.add_argument("--bootstrap-reps",type=int,default=300)
    p.add_argument("--cost-bps",type=float,default=14)
    p.add_argument("--tp-bps",type=float,default=25)
    p.add_argument("--sl-bps",type=float,default=15)
    p.add_argument("--entry-window-minutes",type=int,default=60)
    p.add_argument("--post-entry-minutes",type=int,default=15)
    return p.parse_args()

def safe_json(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def ensure():
    for p in [FEATURE_PATH,EVENT_DATASET_PATH,DETECTED_EVENTS_PATH]:
        if not p.exists(): raise FileNotFoundError(p)
    OUT.mkdir(parents=True,exist_ok=True)

def feature_groups(columns):
    cols=set(columns)
    cat=[c for c in CAT_CANDIDATES if c in cols]
    cal=[c for c in CALENDAR if c in cols]
    dist=[c for c in DISTANCE if c in cols]
    topo=[c for c in TOPOLOGY if c in cols]
    full=list(dict.fromkeys(cat+cal+topo+["current_price"]))
    groups={
      "calendar_only":cal,
      "symbol_only":[c for c in ["symbol"] if c in cols],
      "calendar_symbol":list(dict.fromkeys(cal+[c for c in ["symbol"] if c in cols])),
      "topology_only":list(dict.fromkeys(cat+topo)),
      "distance_only":list(dict.fromkeys([c for c in ["symbol","nearest_side"] if c in cols]+dist)),
      "full":full,
    }
    for name,fs in groups.items():
        bad=[c for c in fs if any(k in c for k in FORBIDDEN)]
        if bad: raise RuntimeError(f"{name}: forbidden features {bad}")
        if not fs: raise RuntimeError(f"{name}: empty feature group")
    return groups

def prep(df,features):
    x=df[features].copy()
    cats=[]
    for c in features:
        if c in CAT_CANDIDATES:
            x[c]=x[c].astype("string").fillna("<MISSING>").astype(str); cats.append(c)
        else:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    return x,cats

def split_times(ts,outer_frac,val_frac,embargo):
    q_outer=ts.quantile(1-outer_frac)
    q_val=ts[ts<q_outer].quantile(1-val_frac/(1-outer_frac))
    return {
      "train_end":q_val-embargo,
      "val_start":q_val+embargo,
      "val_end":q_outer-embargo,
      "outer_start":q_outer+embargo,
      "outer_end":ts.max(),
    }

def sample_train(df,n):
    if len(df)<=n:return df
    return df.iloc[np.linspace(0,len(df)-1,n,dtype=int)].copy()

def fit_model(train,val,features,iters):
    xtr,cats=prep(train,features); xv,_=prep(val,features)
    model=CatBoostClassifier(
      iterations=iters,depth=8,learning_rate=.07,loss_function="MultiClass",
      eval_metric="MultiClass",random_seed=42,verbose=False,
      auto_class_weights="Balanced",allow_writing_files=False,
    )
    model.fit(Pool(xtr,label=train.target_event,cat_features=cats),
              eval_set=Pool(xv,label=val.target_event,cat_features=cats),
              early_stopping_rounds=60,verbose=False)
    return model

def proba3(model,df,features):
    x,cats=prep(df,features)
    raw=model.predict_proba(Pool(x,cat_features=cats))
    out=np.zeros((len(df),3),dtype=float)
    mapping={int(v):i for i,v in enumerate(model.classes_)}
    for cls,src in mapping.items(): out[:,CLASSES.index(cls)]=raw[:,src]
    return out

def class_metrics(y,p):
    pred=np.array(CLASSES)[np.argmax(p,axis=1)]
    return {
      "rows":len(y),"balanced_accuracy":balanced_accuracy_score(y,pred),
      "macro_f1":f1_score(y,pred,average="macro",labels=CLASSES,zero_division=0),
      "log_loss":log_loss(y,p,labels=CLASSES),
      "confusion_matrix":confusion_matrix(y,pred,labels=CLASSES).tolist(),
    }

def add_full_stream_targets(stream,events,window):
    s=stream.sort_values(["symbol","logged_at"]).copy()
    e=events[["symbol","event_time","event_direction"]].copy()
    e=e.rename(columns={"event_time":"next_event_time","event_direction":"next_event_direction"})
    e=e.sort_values(["symbol","next_event_time"])
    joined=pd.merge_asof(
      s,e,left_on="logged_at",right_on="next_event_time",by="symbol",
      direction="forward",allow_exact_matches=False
    )
    delta=joined.next_event_time-joined.logged_at
    joined["target_event"]=np.where(
      delta.notna() & (delta>pd.Timedelta(0)) & (delta<=window),
      joined.next_event_direction,0
    ).astype("int8")
    return joined

def choose_alert_config(val,p):
    event_prob=1-p[:,1]
    direction_conf=np.maximum(p[:,0],p[:,2])/np.maximum(event_prob,1e-12)
    grid=[]
    for q in [.99,.995,.9975]:
      threshold=float(np.quantile(event_prob,q))
      for dc in [.55,.60,.65,.70]:
        mask=(event_prob>=threshold)&(direction_conf>=dc)
        if mask.sum()==0: score=-1
        else:
          pred=np.where(p[mask,2]>=p[mask,0],1,-1)
          truth=val.target_event.to_numpy()[mask]
          precision=float((truth!=0).mean())
          diracc=float((pred[truth!=0]==truth[truth!=0]).mean()) if (truth!=0).any() else 0
          score=precision*diracc*math.log1p(mask.sum())
        grid.append({"q":q,"direction_conf":dc,"threshold":threshold,
                     "alerts":int(mask.sum()),"score":score})
    return max(grid,key=lambda z:z["score"]),grid

def alerts_from_scores(df,p,cfg):
    event_prob=1-p[:,1]
    direction=np.where(p[:,2]>=p[:,0],1,-1)
    conf=np.maximum(p[:,0],p[:,2])/np.maximum(event_prob,1e-12)
    m=(event_prob>=cfg["threshold"])&(conf>=cfg["direction_conf"])
    out=df.loc[m,PRICE_COLS+["target_event"]].copy()
    out["event_probability"]=event_prob[m]
    out["direction_confidence"]=conf[m]
    out["predicted_direction"]=direction[m]
    return out.sort_values(["symbol","logged_at"]).reset_index(drop=True)

def causal_trade(alert, times, prices, start_pos, a):
    side=str(alert.nearest_side)
    if side=="LOWER":
        direction=1; trigger=float(alert.nearest_upper_price)
    elif side=="UPPER":
        direction=-1; trigger=float(alert.nearest_lower_price)
    else:return None
    if not np.isfinite(trigger) or trigger<=0:return None
    start_ns=int(pd.Timestamp(alert.logged_at).value)
    limit_ns=start_ns+int(pd.Timedelta(minutes=a.entry_window_minutes).value)
    j=start_pos+1
    entry=None
    while j<len(times) and times[j]<=limit_ns:
        px=float(prices[j])
        if (direction==1 and px>=trigger) or (direction==-1 and px<=trigger):
            entry=j;break
        j+=1
    if entry is None:return None
    ep=float(prices[entry])
    end_ns=times[entry]+int(pd.Timedelta(minutes=a.post_entry_minutes).value)
    last=np.searchsorted(times,end_ns,side="right")-1
    last=max(entry,min(last,len(times)-1))
    tp=a.tp_bps/10000; sl=a.sl_bps/10000
    exit_i=last; reason="TIME"; gross=direction*(float(prices[last])/ep-1)
    for k in range(entry+1,last+1):
        r=direction*(float(prices[k])/ep-1)
        if r>=tp: exit_i=k;reason="TP";gross=tp;break
        if r<=-sl: exit_i=k;reason="SL";gross=-sl;break
    net=gross-a.cost_bps/10000
    return {
      "signal_id":alert.id,"symbol":alert.symbol,"signal_time":alert.logged_at,
      "entry_time":pd.Timestamp(times[entry],unit="ns",tz="UTC"),
      "exit_time":pd.Timestamp(times[exit_i],unit="ns",tz="UTC"),
      "direction":"LONG" if direction==1 else "SHORT","trigger":trigger,
      "entry_price":ep,"exit_price":float(prices[exit_i]),"exit_reason":reason,
      "gross_bps":gross*10000,"net_bps":net*10000,
    }

def simulate_oracle_free(alerts,stream,a):
    trades=[]
    for sym,g in stream.groupby("symbol",sort=False):
        g=g.sort_values("logged_at")
        times=g.logged_at.astype("int64").to_numpy()
        prices=g.current_price.to_numpy(float)
        ids=g.id.to_numpy()
        pos={v:i for i,v in enumerate(ids)}
        for row in alerts[alerts.symbol.eq(sym)].itertuples(index=False):
            i=pos.get(row.id)
            if i is None: continue
            t=causal_trade(row,times,prices,i,a)
            if t: trades.append(t)
    return pd.DataFrame(trades)

def nonoverlap(df,minutes):
    bucket=df.logged_at.dt.floor(f"{minutes}min")
    return df.assign(_bucket=bucket).sort_values("logged_at").drop_duplicates(["symbol","_bucket"])

def block_bootstrap(df,metric,reps):
    x=df.copy()
    x["_block"]=x.symbol.astype(str)+"|"+x.logged_at.dt.floor("D").astype(str)
    blocks=[g for _,g in x.groupby("_block",sort=False)]
    vals=[]
    for _ in range(reps):
        sample=pd.concat([blocks[i] for i in RNG.integers(0,len(blocks),len(blocks))],ignore_index=True)
        vals.append(metric(sample))
    return {"low":float(np.quantile(vals,.025)),"high":float(np.quantile(vals,.975)),
            "median":float(np.median(vals))}

def main():
    a=args(); ensure(); started=time.time()
    print("Loading squeeze event training data...")
    event=pd.read_parquet(EVENT_DATASET_PATH)
    event["logged_at"]=pd.to_datetime(event.logged_at,utc=True)
    event=event[event.target_event.isin(CLASSES)].sort_values("logged_at").reset_index(drop=True)
    print("Loading full 1h stream...")
    schema=pd.read_parquet(FEATURE_PATH,columns=[]).columns
    groups=feature_groups(schema)
    needed=list(dict.fromkeys(PRICE_COLS+sum(groups.values(),[])))
    stream=pd.read_parquet(FEATURE_PATH,columns=needed,filters=[("timeframe","==","1h")])
    stream["logged_at"]=pd.to_datetime(stream.logged_at,utc=True)
    stream=stream.sort_values(["symbol","logged_at","id"]).drop_duplicates("id").reset_index(drop=True)
    events=pd.read_parquet(DETECTED_EVENTS_PATH)
    events["event_time"]=pd.to_datetime(events.event_time,utc=True)
    stream=add_full_stream_targets(stream,events,pd.Timedelta(minutes=a.event_window_minutes))
    cuts=split_times(event.logged_at,a.outer_fraction,a.validation_fraction,pd.Timedelta(hours=a.embargo_hours))
    train=event[event.logged_at<=cuts["train_end"]].copy()
    val=event[(event.logged_at>=cuts["val_start"])&(event.logged_at<=cuts["val_end"])].copy()
    outer_train=event[event.logged_at<cuts["outer_start"]-pd.Timedelta(hours=a.embargo_hours)].copy()
    outer=stream[(stream.logged_at>=cuts["outer_start"])&(stream.logged_at<=cuts["outer_end"])].copy()
    train=sample_train(train,a.max_train_rows); outer_train=sample_train(outer_train,a.max_train_rows)
    print(f"Train={len(train):,} Val={len(val):,} Outer full stream={len(outer):,}")
    baseline_rows=[]; models={}
    for name,features in groups.items():
        print("Baseline:",name)
        m=fit_model(train,val,features,a.iterations)
        pv=proba3(m,val,features)
        baseline_rows.append({"model":name,"split":"inner_validation",**class_metrics(val.target_event,pv)})
        models[name]=m
    majority=int(train.target_event.value_counts().idxmax())
    pmaj=np.zeros((len(val),3));pmaj[:,CLASSES.index(majority)]=1
    baseline_rows.append({"model":"majority_base_rate","split":"inner_validation",**class_metrics(val.target_event,pmaj)})
    baseline_rows.append({"model":"no_trade","split":"economic_baseline","rows":len(outer),
                          "balanced_accuracy":None,"macro_f1":None,"log_loss":None,
                          "confusion_matrix":None,"net_bps":0.0,"trades":0})
    pd.DataFrame(baseline_rows).to_json(OUT/"baseline_ladder_inner.json",orient="records",indent=2)
    best_features=groups["full"]
    cfg,grid=choose_alert_config(val,proba3(models["full"],val,best_features))
    pd.DataFrame(grid).to_csv(OUT/"inner_grid.csv",index=False)
    print("Frozen config:",cfg)
    final=fit_model(outer_train,val,best_features,a.iterations)
    po=proba3(final,outer,best_features)
    outer_metrics=class_metrics(outer.target_event,po)
    alerts=alerts_from_scores(outer,po,cfg)
    alerts.to_parquet(OUT/"outer_full_stream_alerts.parquet",index=False)
    trades=simulate_oracle_free(alerts,stream,a)
    trades.to_parquet(OUT/"oracle_free_trades.parquet",index=False)
    non=nonoverlap(outer,a.event_window_minutes)
    indicator=(outer.target_event.ne(0)).astype(float)
    rho=float(indicator.autocorr(lag=1)) if len(indicator)>2 else 0.0
    rho=0.0 if not np.isfinite(rho) else max(-.99,min(.99,rho))
    ess=float(len(outer)*(1-rho)/(1+rho))
    pred=np.array(CLASSES)[np.argmax(po,axis=1)]
    evaldf=outer[["symbol","logged_at","target_event"]].copy()
    evaldf["pred"]=pred
    ci=block_bootstrap(evaldf,lambda d: balanced_accuracy_score(d.target_event,d.pred),a.bootstrap_reps)
    econ={
      "signals":int(len(alerts)),"fills":int(len(trades)),
      "fill_rate":float(len(trades)/len(alerts)) if len(alerts) else None,
      "mean_net_bps":float(trades.net_bps.mean()) if len(trades) else None,
      "median_net_bps":float(trades.net_bps.median()) if len(trades) else None,
      "total_net_bps":float(trades.net_bps.sum()) if len(trades) else 0.0,
      "win_rate":float((trades.net_bps>0).mean()) if len(trades) else None,
    }
    report={
      "status":"research_only","forbidden_execution_columns":list(FORBIDDEN),
      "split":cuts,"frozen_config":cfg,"outer_metrics":outer_metrics,
      "economic_oracle_free":econ,
      "sample_size":{
        "nominal_rows":int(len(outer)),"non_overlapping_rows":int(len(non)),
        "lag1_event_autocorrelation":rho,"effective_sample_size_estimate":ess,
        "block_bootstrap_balanced_accuracy_95pct":ci,
      },
      "topology_increment_vs_calendar_symbol":{
        "inner_balanced_accuracy": next(r["balanced_accuracy"] for r in baseline_rows if r["model"]=="topology_only")
          - next(r["balanced_accuracy"] for r in baseline_rows if r["model"]=="calendar_symbol"),
        "full_increment_vs_calendar_symbol": next(r["balanced_accuracy"] for r in baseline_rows if r["model"]=="full")
          - next(r["balanced_accuracy"] for r in baseline_rows if r["model"]=="calendar_symbol"),
      },
      "runtime_seconds":time.time()-started,
    }
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=safe_json),encoding="utf-8")
    print(json.dumps(report,indent=2,default=safe_json))
    print("Done:",OUT/"report.json")

if __name__=="__main__":
    main()
