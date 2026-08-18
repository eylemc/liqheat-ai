from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

ROOT = Path('data/research/bias_v2_pressure')
PRESSURE = Path('data/research/liquidation_pressure/local_historical_pressure_features.parquet')
OUT = ROOT / 'pressure_regime_analysis'
HORIZONS = ('1h', '4h')

PRESSURE_COLS = [
    'snapshot_id','signed_pressure','liquidation_pressure_score','direction_confidence',
    'mean_30m','mean_60m','mean_90m','mean_120m',
    'persistence_30m','persistence_60m','persistence_90m','persistence_120m',
    'flips_30m','flips_60m','flips_90m','flips_120m',
    'slope_30m_per_min','slope_60m_per_min','slope_90m_per_min','slope_120m_per_min',
    'acceleration_30_vs_120',
]


def safe_metrics(d: pd.DataFrame) -> dict:
    if d.empty:
        return {'n': 0}
    y = d['target'].to_numpy(dtype=int)
    p = d['p_baseline_upper'].to_numpy(dtype=float)
    pred = (p >= 0.5).astype(int)
    result = {
        'n': int(len(d)),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'log_loss': float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
        'mean_confidence': float(np.maximum(p, 1-p).mean()),
    }
    if len(np.unique(y)) == 2:
        result['balanced_accuracy'] = float(balanced_accuracy_score(y, pred))
    else:
        result['balanced_accuracy'] = None
    return result


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x['baseline_pred'] = np.where(x['p_baseline_upper'] >= 0.5, 'UPPER_FIRST', 'LOWER_FIRST')
    x['baseline_conf'] = np.maximum(x['p_baseline_upper'], 1-x['p_baseline_upper'])
    x['pressure_dir_now'] = np.where(x['signed_pressure'] > 0, 'UPPER_FIRST', np.where(x['signed_pressure'] < 0, 'LOWER_FIRST', 'TIE'))
    x['pressure_dir_120m'] = np.where(x['mean_120m'] > 0, 'UPPER_FIRST', np.where(x['mean_120m'] < 0, 'LOWER_FIRST', 'TIE'))
    x['pressure_agrees_now'] = x['pressure_dir_now'].eq(x['baseline_pred'])
    x['pressure_agrees_120m'] = x['pressure_dir_120m'].eq(x['baseline_pred'])
    x['abs_mean_120m_signed'] = x['mean_120m'].abs()
    x['abs_slope_120m'] = x['slope_120m_per_min'].abs()
    x['abs_accel'] = x['acceleration_30_vs_120'].abs()
    return x


def record(rows: list[dict], horizon: str, symbol: str, name: str, mask: pd.Series, frame: pd.DataFrame, overall_acc: float):
    s = frame.loc[mask].copy()
    m = safe_metrics(s)
    row = {
        'horizon': horizon,
        'symbol': symbol,
        'regime': name,
        'coverage': float(len(s) / len(frame)) if len(frame) else 0.0,
        'accuracy_delta_vs_overall': (m.get('accuracy') - overall_acc) if m.get('n',0) else None,
        **m,
    }
    rows.append(row)


def analyse_one(horizon: str) -> tuple[pd.DataFrame, dict]:
    pred_path = ROOT / f'holdout_predictions_{horizon}.parquet'
    pred = pd.read_parquet(pred_path)
    pr = pd.read_parquet(PRESSURE, columns=PRESSURE_COLS).rename(columns={'snapshot_id':'id'})
    d = pred.merge(pr, on='id', how='inner', validate='one_to_one')
    d = add_derived(d)

    rows: list[dict] = []
    overall = safe_metrics(d)
    overall_acc = overall['accuracy']

    def run(frame: pd.DataFrame, symbol: str):
        base = safe_metrics(frame)
        acc = base['accuracy']
        record(rows,horizon,symbol,'ALL',pd.Series(True,index=frame.index),frame,acc)

        for thr in (60,70,80,90):
            record(rows,horizon,symbol,f'PERSIST120_GE_{thr}',frame['persistence_120m']>=thr,frame,acc)
            record(rows,horizon,symbol,f'PERSIST120_GE_{thr}_AGREE', (frame['persistence_120m']>=thr)&frame['pressure_agrees_120m'],frame,acc)
            record(rows,horizon,symbol,f'PERSIST120_GE_{thr}_DIVERGE',(frame['persistence_120m']>=thr)&(~frame['pressure_agrees_120m']),frame,acc)

        for flips in (2,4,6,10):
            record(rows,horizon,symbol,f'FLIPS120_LE_{flips}',frame['flips_120m']<=flips,frame,acc)
            record(rows,horizon,symbol,f'FLIPS120_LE_{flips}_AGREE',(frame['flips_120m']<=flips)&frame['pressure_agrees_120m'],frame,acc)

        for conf in (.55,.60,.65,.70):
            record(rows,horizon,symbol,f'BASECONF_GE_{conf:.2f}',frame['baseline_conf']>=conf,frame,acc)
            record(rows,horizon,symbol,f'BASECONF_GE_{conf:.2f}_PRESSURE_AGREE',(frame['baseline_conf']>=conf)&frame['pressure_agrees_120m'],frame,acc)
            record(rows,horizon,symbol,f'BASECONF_GE_{conf:.2f}_PRESSURE_DIVERGE',(frame['baseline_conf']>=conf)&(~frame['pressure_agrees_120m']),frame,acc)

        # Quantile-derived strength regimes are calculated only from this untouched holdout
        # for descriptive/exploratory analysis; they are NOT deployment thresholds.
        for col,label in [('abs_mean_120m_signed','ABS_MEAN120'),('abs_slope_120m','ABS_SLOPE120'),('abs_accel','ABS_ACCEL')]:
            valid = frame[col].dropna()
            if valid.empty: continue
            for q in (.50,.75,.90):
                cut=float(valid.quantile(q))
                mask=frame[col]>=cut
                record(rows,horizon,symbol,f'{label}_GE_Q{int(q*100)}',mask,frame,acc)
                record(rows,horizon,symbol,f'{label}_GE_Q{int(q*100)}_AGREE',mask&frame['pressure_agrees_120m'],frame,acc)
                record(rows,horizon,symbol,f'{label}_GE_Q{int(q*100)}_DIVERGE',mask&(~frame['pressure_agrees_120m']),frame,acc)

        # Combined regimes we actually hypothesized before looking at results.
        stable=(frame['persistence_120m']>=70)&(frame['flips_120m']<=6)
        strong=(frame['abs_mean_120m_signed']>=frame['abs_mean_120m_signed'].quantile(.75))
        record(rows,horizon,symbol,'STABLE_PRESSURE',stable,frame,acc)
        record(rows,horizon,symbol,'STABLE_PRESSURE_AGREE',stable&frame['pressure_agrees_120m'],frame,acc)
        record(rows,horizon,symbol,'STABLE_PRESSURE_DIVERGE',stable&(~frame['pressure_agrees_120m']),frame,acc)
        record(rows,horizon,symbol,'STABLE_STRONG_PRESSURE',stable&strong,frame,acc)
        record(rows,horizon,symbol,'STABLE_STRONG_PRESSURE_AGREE',stable&strong&frame['pressure_agrees_120m'],frame,acc)
        record(rows,horizon,symbol,'STABLE_STRONG_PRESSURE_DIVERGE',stable&strong&(~frame['pressure_agrees_120m']),frame,acc)

    run(d,'ALL')
    for symbol, g in d.groupby('symbol', sort=True):
        run(g, str(symbol))

    result=pd.DataFrame(rows)
    summary={
        'horizon':horizon,
        'joined_holdout_rows':int(len(d)),
        'overall':overall,
        'top_regimes_min_5pct_coverage': result[(result.symbol=='ALL')&(result.coverage>=.05)&(result.regime!='ALL')]
            .sort_values(['accuracy_delta_vs_overall','coverage'],ascending=[False,False])
            .head(15)[['regime','n','coverage','accuracy','balanced_accuracy','accuracy_delta_vs_overall','mean_confidence']]
            .to_dict('records'),
        'worst_regimes_min_5pct_coverage': result[(result.symbol=='ALL')&(result.coverage>=.05)&(result.regime!='ALL')]
            .sort_values(['accuracy_delta_vs_overall','coverage'],ascending=[True,False])
            .head(15)[['regime','n','coverage','accuracy','balanced_accuracy','accuracy_delta_vs_overall','mean_confidence']]
            .to_dict('records'),
    }
    return result, summary


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]; summaries={}
    for h in HORIZONS:
        print(f'\n=== {h.upper()} PRESSURE REGIME ANALYSIS ===')
        res,summary=analyse_one(h)
        all_rows.append(res); summaries[h]=summary
        print('Overall:', json.dumps(summary['overall'],indent=2))
        print('\nTop regimes (>=5% coverage):')
        for r in summary['top_regimes_min_5pct_coverage'][:10]:
            print(f"{r['regime']:<38} cov={r['coverage']:.1%} acc={r['accuracy']:.4f} delta={r['accuracy_delta_vs_overall']:+.4f}")
    out=pd.concat(all_rows,ignore_index=True)
    out.to_csv(OUT/'regime_metrics.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print('\nSaved:',OUT)
    print('NOTE: quantile regimes are exploratory. Any promising threshold must be frozen on development data and retested on a separate untouched period before production use.')

if __name__=='__main__':
    main()
