from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

TOPOLOGY = Path('data/features/liq_topology_v2_ml_features.parquet')
PRESSURE = Path('data/research/liquidation_pressure/local_historical_pressure_features.parquet')
LABELS = Path('data/features/liq_topology_v2_sweep_labels.parquet')
OUT = Path('data/research/bias_v2_pressure')
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT']
TF = '24h'

BASE = [
 'upper_distance_pct','lower_distance_pct','distance_advantage','pool_volume_ratio',
 'distance_pressure_ratio','topology_imbalance','nearest_side_code','active_level_difference',
 'active_level_total','total_volume_imbalance_check','signed_distance_edge',
 'log1p_upper_pool_volume','log1p_lower_pool_volume','log1p_upper_total_volume',
 'log1p_lower_total_volume','hour_sin','hour_cos','dow_sin','dow_cos'
]
PRESS = []
for m in (30,60,90,120):
    PRESS += [f'mean_{m}m',f'abs_mean_{m}m',f'peak_abs_{m}m',f'sample_count_{m}m',
              f'persistence_{m}m',f'flips_{m}m',f'slope_{m}m_per_min']
PRESS += ['acceleration_30_vs_120','liquidation_pressure_score','direction_confidence','signed_pressure']


def metrics(y, p):
    pred=(p>=0.5).astype(np.int8)
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'log_loss':float(log_loss(y,np.c_[1-p,p],labels=[0,1]))}


def fit_predict(train, test, features, seed):
    model=CatBoostClassifier(iterations=500,depth=7,learning_rate=.06,loss_function='Logloss',
        eval_metric='Logloss',random_seed=seed,verbose=False,thread_count=-1,l2_leaf_reg=5,
        allow_writing_files=False)
    model.fit(train[features],train['target'])
    return model.predict_proba(test[features])[:,1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--horizons',nargs='+',choices=['1h','4h'],default=['1h','4h'])
    ap.add_argument('--folds',type=int,default=4)
    ap.add_argument('--holdout-frac',type=float,default=.20)
    a=ap.parse_args(); t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)

    print('Loading topology...')
    topo=pd.read_parquet(TOPOLOGY,columns=['id','logged_at','symbol','timeframe']+BASE)
    topo['symbol']=topo.symbol.astype(str); topo['timeframe']=topo.timeframe.astype(str)
    topo=topo[topo.symbol.isin(SYMBOLS)&topo.timeframe.eq(TF)].copy()
    print('Loading pressure...')
    pr=pd.read_parquet(PRESSURE,columns=['snapshot_id']+PRESS)
    pr=pr.rename(columns={'snapshot_id':'id'})
    print('Loading sweep labels...')
    label_cols=['id']
    for h in a.horizons: label_cols += [f'sweep_valid_{h}',f'sweep_label_{h}']
    lab=pd.read_parquet(LABELS,columns=label_cols)
    df=topo.merge(pr,on='id',how='inner',validate='one_to_one').merge(lab,on='id',how='inner',validate='one_to_one')
    df['logged_at']=pd.to_datetime(df.logged_at,utc=True); df=df.sort_values('logged_at').reset_index(drop=True)
    print(f'Joined rows: {len(df):,}  {df.logged_at.min()} -> {df.logged_at.max()}')

    all_results=[]
    for horizon in a.horizons:
        label=f'sweep_label_{horizon}'; valid=f'sweep_valid_{horizon}'
        d=df[(df[valid].astype(bool)) & df[label].isin(['UPPER_FIRST','LOWER_FIRST'])].copy()
        d['target']=(d[label]=='UPPER_FIRST').astype(np.int8)
        # complete-case parity: baseline and candidate are evaluated on identical observations
        features=BASE+PRESS
        d=d.dropna(subset=features+['target']).sort_values('logged_at').reset_index(drop=True)
        n=len(d); hold_start=int(n*(1-a.holdout_frac)); dev=d.iloc[:hold_start].copy(); hold=d.iloc[hold_start:].copy()
        print(f'\n=== {horizon} === rows={n:,} dev={len(dev):,} untouched_holdout={len(hold):,}')

        # Expanding chronological folds over development period.
        boundaries=np.linspace(0,len(dev),a.folds+2,dtype=int)
        for fold in range(a.folds):
            tr_end=boundaries[fold+1]; te_end=boundaries[fold+2]
            train=dev.iloc[:tr_end]; test=dev.iloc[tr_end:te_end]
            if len(train)<10000 or len(test)==0: continue
            for name,fs in [('baseline',BASE),('pressure',features)]:
                p=fit_predict(train,test,fs,1000+fold)
                r=metrics(test.target.to_numpy(),p); r.update(horizon=horizon,split='walk_forward',fold=fold+1,model=name,
                    train_first=str(train.logged_at.min()),train_last=str(train.logged_at.max()),test_first=str(test.logged_at.min()),test_last=str(test.logged_at.max()))
                all_results.append(r)
                print(f'fold {fold+1} {name:8s} BA={r["balanced_accuracy"]:.4f} F1={r["macro_f1"]:.4f} LL={r["log_loss"]:.4f}')

        # Final untouched chronological holdout; train only on pre-holdout history.
        hold_preds={}
        for name,fs in [('baseline',BASE),('pressure',features)]:
            p=fit_predict(dev,hold,fs,9000)
            hold_preds[name]=p
            r=metrics(hold.target.to_numpy(),p); r.update(horizon=horizon,split='untouched_holdout',fold=0,model=name,
                train_first=str(dev.logged_at.min()),train_last=str(dev.logged_at.max()),test_first=str(hold.logged_at.min()),test_last=str(hold.logged_at.max()))
            all_results.append(r)
            print(f'HOLDOUT {name:8s} BA={r["balanced_accuracy"]:.4f} F1={r["macro_f1"]:.4f} LL={r["log_loss"]:.4f}')
        hp=hold[['id','logged_at','symbol',label,'target']].copy(); hp['p_baseline_upper']=hold_preds['baseline']; hp['p_pressure_upper']=hold_preds['pressure']
        hp.to_parquet(OUT/f'holdout_predictions_{horizon}.parquet',index=False,compression='zstd')

    res=pd.DataFrame(all_results); res.to_csv(OUT/'metrics.csv',index=False)
    summary={}
    for h in a.horizons:
        sub=res[(res.horizon==h)&(res.split=='walk_forward')]
        hld=res[(res.horizon==h)&(res.split=='untouched_holdout')]
        summary[h]={'walk_forward_mean':{},'untouched_holdout':{}}
        for name in ['baseline','pressure']:
            s=sub[sub.model==name]; q=hld[hld.model==name]
            summary[h]['walk_forward_mean'][name]={k:float(s[k].mean()) for k in ['accuracy','balanced_accuracy','macro_f1','log_loss']}
            summary[h]['untouched_holdout'][name]={k:float(q.iloc[0][k]) for k in ['accuracy','balanced_accuracy','macro_f1','log_loss']} if len(q) else {}
        if all(x in summary[h]['untouched_holdout'] for x in ['baseline','pressure']):
            b=summary[h]['untouched_holdout']['baseline']; p=summary[h]['untouched_holdout']['pressure']
            summary[h]['holdout_delta_pressure_minus_baseline']={
                'accuracy':p['accuracy']-b['accuracy'],'balanced_accuracy':p['balanced_accuracy']-b['balanced_accuracy'],
                'macro_f1':p['macro_f1']-b['macro_f1'],'log_loss':p['log_loss']-b['log_loss']}
    summary['warning']='Historical pressure uses the CURRENT production squeeze model over past features. Treat as research evidence until squeeze-model training chronology is audited for look-ahead leakage.'
    summary['elapsed_seconds']=time.time()-t0
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n=== SUMMARY ==='); print(json.dumps(summary,indent=2)); print('\nSaved:',OUT)

if __name__=='__main__': main()
