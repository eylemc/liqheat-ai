#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss

FEATURE_PATH = Path('data/features/liq_topology_v2_ml_features.parquet')
OUT = Path('data/reports/position_guardian_v3')
MODEL_OUT = Path('data/models/position_guardian_v3')
HORIZONS = [15, 30, 60]
SIDES = ['LONG', 'SHORT']
CAT = ['symbol', 'timeframe', 'nearest_side']
FEATURES = [
    'symbol','timeframe','nearest_side','hour_sin','hour_cos','dow_sin','dow_cos','is_weekend_utc','current_price',
    'has_upper_level','has_lower_level','has_topology','nearest_side_code','upper_distance_pct','lower_distance_pct',
    'distance_advantage','signed_distance_edge','log1p_upper_distance_pct','log1p_lower_distance_pct','log1p_distance_advantage',
    'upper_pool_volume','lower_pool_volume','nearest_pool_volume','farther_pool_volume','upper_total_volume','lower_total_volume',
    'upper_active_levels','lower_active_levels','log1p_upper_pool_volume','log1p_lower_pool_volume','log1p_nearest_pool_volume',
    'log1p_farther_pool_volume','log1p_upper_total_volume','log1p_lower_total_volume','log1p_upper_active_levels',
    'log1p_lower_active_levels','pool_volume_ratio','log1p_pool_volume_ratio','distance_pressure_ratio',
    'log1p_distance_pressure_ratio','topology_imbalance','total_volume_imbalance_check','active_level_difference','active_level_total'
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--sample-every-minutes', type=int, default=15)
    p.add_argument('--max-train-rows', type=int, default=500000)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--validation-fraction', type=float, default=.15)
    p.add_argument('--test-fraction', type=float, default=.15)
    p.add_argument('--embargo-hours', type=float, default=4)
    p.add_argument('--exit-stop-bps', type=float, default=15)
    p.add_argument('--recovery-bps', type=float, default=12)
    p.add_argument('--endpoint-exit-bps', type=float, default=-8)
    p.add_argument('--min-precision', type=float, default=.45)
    p.add_argument('--min-coverage', type=float, default=.05)
    p.add_argument('--min-alerts', type=int, default=100)
    return p.parse_args()


def prep(df):
    x = df[FEATURES].copy()
    cats=[]
    for c in FEATURES:
        if c in CAT:
            x[c]=x[c].astype('string').fillna('<MISSING>').astype(str); cats.append(c)
        else:
            x[c]=pd.to_numeric(x[c],errors='coerce')
    return x,cats


def load_stream(a):
    schema=set(pq.read_schema(FEATURE_PATH).names)
    missing=[c for c in FEATURES+['id','logged_at'] if c not in schema]
    if missing: raise RuntimeError(f'Missing columns: {missing}')
    cols=list(dict.fromkeys(['id','logged_at']+FEATURES))
    df=pd.read_parquet(FEATURE_PATH,columns=cols,filters=[('timeframe','==',a.timeframe)])
    df['logged_at']=pd.to_datetime(df.logged_at,utc=True).astype('datetime64[ns, UTC]')
    df['symbol']=df.symbol.astype('string')
    df=df.sort_values(['symbol','logged_at','id']).drop_duplicates('id').reset_index(drop=True)
    if a.sample_every_minutes>0:
        df['_bucket']=df.logged_at.dt.floor(f'{a.sample_every_minutes}min')
        df=df.drop_duplicates(['symbol','_bucket']).drop(columns='_bucket').reset_index(drop=True)
    return df


def add_outcomes(df,horizon,side,a):
    pieces=[]
    horizon_ns=int(pd.Timedelta(minutes=horizon).value)
    sign=1.0 if side=='LONG' else -1.0
    for sym,g in df.groupby('symbol',sort=False):
        g=g.sort_values('logged_at').copy()
        t=g.logged_at.astype('int64').to_numpy(); p=g.current_price.to_numpy(float)
        n=len(g); y=np.full(n,np.nan); mfe=np.full(n,np.nan); mae=np.full(n,np.nan); end=np.full(n,np.nan)
        for i in range(n):
            j=np.searchsorted(t,t[i]+horizon_ns,side='right')
            if j<=i+1: continue
            path=sign*(p[i+1:j]/p[i]-1)*10000
            mfe[i]=np.nanmax(path); mae[i]=np.nanmin(path); end[i]=path[-1]
            y[i]=int((mae[i] <= -a.exit_stop_bps and mfe[i] < a.recovery_bps) or (end[i] <= a.endpoint_exit_bps))
        g['exit_risk']=y; g['mfe_bps']=mfe; g['mae_bps']=mae; g['endpoint_bps']=end
        pieces.append(g)
    out=pd.concat(pieces,ignore_index=True)
    return out.dropna(subset=['exit_risk']).assign(exit_risk=lambda x:x.exit_risk.astype('int8'))


def split(df,a):
    ts=df.logged_at
    test_start=ts.quantile(1-a.test_fraction)
    val_start=ts[ts<test_start].quantile(1-a.validation_fraction/(1-a.test_fraction))
    e=pd.Timedelta(hours=a.embargo_hours)
    tr=df[df.logged_at<=val_start-e].copy()
    va=df[(df.logged_at>=val_start+e)&(df.logged_at<=test_start-e)].copy()
    te=df[df.logged_at>=test_start+e].copy()
    if len(tr)>a.max_train_rows:
        tr=tr.iloc[np.linspace(0,len(tr)-1,a.max_train_rows,dtype=int)]
    return tr,va,te,{'train_end':str(val_start-e),'validation_start':str(val_start+e),'test_start':str(test_start+e),'test_end':str(ts.max())}


def fit(tr,va,a):
    xtr,cats=prep(tr); xv,_=prep(va)
    m=CatBoostClassifier(iterations=a.iterations,depth=8,learning_rate=.06,loss_function='Logloss',eval_metric='AUC',
        random_seed=42,verbose=False,auto_class_weights='Balanced',allow_writing_files=False)
    m.fit(Pool(xtr,label=tr.exit_risk,cat_features=cats),eval_set=Pool(xv,label=va.exit_risk,cat_features=cats),early_stopping_rounds=60,verbose=False)
    return m


def probs(m,df):
    x,cats=prep(df); return m.predict_proba(Pool(x,cat_features=cats))[:,1]


def regime_masks(df,side):
    nearest=df.nearest_side.astype('string')
    if side=='LONG': adverse_side=nearest.eq('LOWER')
    else: adverse_side=nearest.eq('UPPER')
    imbal=pd.to_numeric(df.topology_imbalance,errors='coerce').fillna(0)
    signed=pd.to_numeric(df.signed_distance_edge,errors='coerce').fillna(0)
    adverse_imbalance=(imbal<0) if side=='LONG' else (imbal>0)
    adverse_distance=(signed<0) if side=='LONG' else (signed>0)

    adverse_side_np = adverse_side.fillna(False).to_numpy(dtype=bool)
    adverse_imbalance_np = adverse_imbalance.fillna(False).to_numpy(dtype=bool)
    adverse_distance_np = adverse_distance.fillna(False).to_numpy(dtype=bool)
    vote_count = (
        adverse_side_np.astype(np.int8)
        + adverse_imbalance_np.astype(np.int8)
        + adverse_distance_np.astype(np.int8)
    )

    return {
        'all':np.ones(len(df),dtype=bool),
        'adverse_nearest_side':adverse_side_np,
        'adverse_imbalance':adverse_imbalance_np,
        'adverse_distance_edge':adverse_distance_np,
        'two_of_three':vote_count>=2,
    }


def policy_grid(df,p,side,a):
    masks=regime_masks(df,side); rows=[]
    for regime,rm in masks.items():
        for th in np.arange(.40,.91,.025):
            pred=(p>=th)&rm; n=int(pred.sum()); cov=n/len(df)
            if n:
                prec=float(precision_score(df.exit_risk,pred,zero_division=0)); rec=float(recall_score(df.exit_risk,pred,zero_division=0))
                lift=prec/max(float(df.exit_risk.mean()),1e-9)
            else: prec=rec=lift=0.0
            eligible=n>=a.min_alerts and cov>=a.min_coverage and prec>=a.min_precision
            score=(prec*2+rec+min(cov,.25)) if eligible else -1
            rows.append({'regime':regime,'threshold':round(float(th),3),'alerts':n,'coverage':cov,'precision':prec,'recall':rec,'lift_vs_base':lift,'eligible':eligible,'score':score})
    eligible=[r for r in rows if r['eligible']]
    if eligible: chosen=max(eligible,key=lambda r:(r['score'],r['precision'],r['recall']))
    else:
        fallback=[r for r in rows if r['alerts']>=max(25,a.min_alerts//4)]
        chosen=max(fallback,key=lambda r:(r['precision'],r['recall'])) if fallback else max(rows,key=lambda r:r['alerts'])
        chosen=dict(chosen); chosen['fallback']=True
    return chosen,rows


def evaluate(df,p,side,policy):
    rm=regime_masks(df,side)[policy['regime']]
    pred=(p>=policy['threshold'])&rm
    base=float(df.exit_risk.mean()); n=int(pred.sum())
    precision=float(precision_score(df.exit_risk,pred,zero_division=0))
    return {
        'rows':len(df),'base_exit_risk_rate':base,'alerts':n,'coverage':n/len(df),
        'precision':precision,
        'recall':float(recall_score(df.exit_risk,pred,zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(df.exit_risk,pred)),
        'lift_vs_base':float(precision/base) if base>0 else None,
        'roc_auc':float(roc_auc_score(df.exit_risk,p)),'pr_auc':float(average_precision_score(df.exit_risk,p)),
        'brier':float(brier_score_loss(df.exit_risk,p)),
    }


def main():
    a=parse_args(); OUT.mkdir(parents=True,exist_ok=True); MODEL_OUT.mkdir(parents=True,exist_ok=True)
    stream=load_stream(a); reports=[]
    for side in SIDES:
      for h in HORIZONS:
        print(f'{side} {h}m')
        ds=add_outcomes(stream,h,side,a); tr,va,te,cuts=split(ds,a)
        m=fit(tr,va,a); pv=probs(m,va); pt=probs(m,te)
        policy,grid=policy_grid(va,pv,side,a); metrics=evaluate(te,pt,side,policy)
        key=f'{side.lower()}_{h}m'; d=MODEL_OUT/key; d.mkdir(parents=True,exist_ok=True); m.save_model(str(d/'model.cbm'))
        pred=te[['id','logged_at','symbol','current_price','exit_risk','mfe_bps','mae_bps','endpoint_bps']].copy(); pred['p_exit_risk']=pt
        pred['alert']=((pt>=policy['threshold'])&regime_masks(te,side)[policy['regime']]).astype('int8')
        pred.to_parquet(OUT/f'{key}_test_predictions.parquet',index=False)
        report={'side':side,'horizon_minutes':h,'split':cuts,'counts':{'dataset':len(ds),'train':len(tr),'validation':len(va),'test':len(te),'validation_base_rate':float(va.exit_risk.mean()),'test_base_rate':float(te.exit_risk.mean())},'selected_policy':policy,'test_metrics':metrics,'validation_grid':grid}
        (OUT/f'{key}_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8'); reports.append(report)
    summary={'status':'research_only','target':'HIGH_PRECISION_EXIT_RISK','selection':'validation-only threshold + causal regime filter','reports':reports}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('Done:',OUT/'summary.json')

if __name__=='__main__': main()
