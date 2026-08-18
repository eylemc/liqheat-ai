from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

ROOT = Path('data/research/bias_v2_pressure')
PRESSURE = Path('data/research/liquidation_pressure/local_historical_pressure_features.parquet')
OUT = ROOT / 'pressure_frozen_validation'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
CONF_THRESHOLD = 0.70


def score_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {'n': 0}
    y = df['target'].to_numpy(dtype=int)
    p = df['p_baseline_upper'].to_numpy(dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        'n': int(len(df)),
        'coverage': None,
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'log_loss': float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
        'mean_confidence': float(np.maximum(p, 1-p).mean()),
    }


def add_pressure_state(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['baseline_pred'] = (d['p_baseline_upper'] >= 0.5).astype(np.int8)
    d['baseline_confidence'] = np.maximum(d['p_baseline_upper'], 1-d['p_baseline_upper'])
    d['pressure_pred'] = (d['signed_pressure'] > 0).astype(np.int8)
    d['pressure_agree'] = d['baseline_pred'].eq(d['pressure_pred'])
    d['high_conf'] = d['baseline_confidence'] >= CONF_THRESHOLD
    d['rule'] = np.select(
        [d['high_conf'] & d['pressure_agree'], d['high_conf'] & ~d['pressure_agree'], d['high_conf']],
        ['HIGHCONF_AGREE','HIGHCONF_DIVERGE','HIGHCONF'],
        default='OTHER'
    )
    return d


def evaluate_slice(df: pd.DataFrame, horizon: str, dimension: str, bucket: str) -> list[dict]:
    rows=[]
    overall = score_frame(df)
    total = len(df)
    for rule, mask in [
        ('HIGHCONF', df['high_conf']),
        ('HIGHCONF_AGREE', df['high_conf'] & df['pressure_agree']),
        ('HIGHCONF_DIVERGE', df['high_conf'] & ~df['pressure_agree']),
    ]:
        sub=df.loc[mask]
        m=score_frame(sub)
        m.update({
            'horizon':horizon,'dimension':dimension,'bucket':bucket,'rule':rule,
            'coverage': float(len(sub)/total) if total else 0.0,
            'overall_accuracy': overall.get('accuracy'),
            'delta_vs_overall_accuracy': (m.get('accuracy')-overall.get('accuracy')) if m.get('n',0) and overall.get('accuracy') is not None else None,
        })
        rows.append(m)
    return rows


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--horizons', nargs='+', choices=['1h','4h'], default=['1h','4h'])
    ap.add_argument('--blocks', type=int, default=4)
    args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pressure = pd.read_parquet(PRESSURE, columns=['snapshot_id','signed_pressure'])
    pressure = pressure.rename(columns={'snapshot_id':'id'})
    results=[]
    summary={}

    for horizon in args.horizons:
        pred_path = ROOT / f'holdout_predictions_{horizon}.parquet'
        pred = pd.read_parquet(pred_path)
        pred['logged_at']=pd.to_datetime(pred['logged_at'], utc=True)
        d=pred.merge(pressure,on='id',how='left',validate='one_to_one').dropna(subset=['signed_pressure'])
        d=add_pressure_state(d).sort_values('logged_at').reset_index(drop=True)

        # Global holdout reference
        results += evaluate_slice(d,horizon,'all','ALL')

        # Asset robustness
        for symbol in SYMBOLS:
            results += evaluate_slice(d[d.symbol.eq(symbol)],horizon,'symbol',symbol)

        # Chronological robustness across equal-count blocks
        idx_blocks=np.array_split(np.arange(len(d)), args.blocks)
        for i, idx in enumerate(idx_blocks, start=1):
            b=d.iloc[idx]
            bucket=f'BLOCK_{i}_{b.logged_at.min().date()}_{b.logged_at.max().date()}'
            results += evaluate_slice(b,horizon,'time_block',bucket)

        # Symbol x time-block robustness
        for symbol in SYMBOLS:
            ds=d[d.symbol.eq(symbol)].reset_index(drop=True)
            for i, idx in enumerate(np.array_split(np.arange(len(ds)), args.blocks), start=1):
                b=ds.iloc[idx]
                if b.empty: continue
                bucket=f'{symbol}_B{i}_{b.logged_at.min().date()}_{b.logged_at.max().date()}'
                results += evaluate_slice(b,horizon,'symbol_time',bucket)

        hi=d[d.high_conf]
        ag=hi[hi.pressure_agree]
        dv=hi[~hi.pressure_agree]
        summary[horizon]={
            'frozen_conf_threshold':CONF_THRESHOLD,
            'holdout_rows':int(len(d)),
            'highconf_rows':int(len(hi)),
            'highconf_accuracy':score_frame(hi).get('accuracy'),
            'agree_rows':int(len(ag)),
            'agree_accuracy':score_frame(ag).get('accuracy'),
            'diverge_rows':int(len(dv)),
            'diverge_accuracy':score_frame(dv).get('accuracy'),
            'agree_minus_highconf_pp': (score_frame(ag).get('accuracy')-score_frame(hi).get('accuracy'))*100 if len(ag) and len(hi) else None,
            'diverge_minus_highconf_pp': (score_frame(dv).get('accuracy')-score_frame(hi).get('accuracy'))*100 if len(dv) and len(hi) else None,
        }

    res=pd.DataFrame(results)
    res.to_csv(OUT/'frozen_validation_metrics.csv',index=False)

    # Compact robustness summary for the exact frozen confirmation rule.
    robust=[]
    for horizon in args.horizons:
        sub=res[(res.horizon.eq(horizon)) & (res.rule.isin(['HIGHCONF','HIGHCONF_AGREE']))]
        pivot=sub.pivot_table(index=['dimension','bucket'],columns='rule',values='accuracy',aggfunc='first').reset_index()
        if {'HIGHCONF','HIGHCONF_AGREE'}.issubset(pivot.columns):
            pivot['agree_minus_highconf_pp']=(pivot['HIGHCONF_AGREE']-pivot['HIGHCONF'])*100
            for _,r in pivot.iterrows():
                robust.append({'horizon':horizon,'dimension':r['dimension'],'bucket':r['bucket'],
                    'highconf_accuracy':float(r['HIGHCONF']),'agree_accuracy':float(r['HIGHCONF_AGREE']),
                    'agree_minus_highconf_pp':float(r['agree_minus_highconf_pp'])})
    robust_df=pd.DataFrame(robust)
    robust_df.to_csv(OUT/'agreement_robustness.csv',index=False)

    for horizon in args.horizons:
        s=robust_df[robust_df.horizon.eq(horizon)]
        if not s.empty:
            summary[horizon]['robustness']={
                'slices':int(len(s)),
                'positive_slices':int((s.agree_minus_highconf_pp>0).sum()),
                'negative_slices':int((s.agree_minus_highconf_pp<0).sum()),
                'median_gain_pp':float(s.agree_minus_highconf_pp.median()),
                'mean_gain_pp':float(s.agree_minus_highconf_pp.mean()),
                'min_gain_pp':float(s.agree_minus_highconf_pp.min()),
                'max_gain_pp':float(s.agree_minus_highconf_pp.max()),
            }

    summary['method_note']=(
        'Frozen rule: baseline confidence >= 0.70; pressure agreement is sign(signed_pressure) '
        'matching baseline UPPER/LOWER prediction. No threshold search is performed in this script. '
        'This reuses the prior untouched holdout, so it is robustness analysis rather than a new independent holdout.'
    )
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    print(json.dumps(summary,indent=2))
    print('\n=== AGREEMENT ROBUSTNESS ===')
    if not robust_df.empty:
        for horizon in args.horizons:
            print(f'\n{horizon.upper()}')
            s=robust_df[robust_df.horizon.eq(horizon)].sort_values(['dimension','bucket'])
            for _,r in s.iterrows():
                print(f"{r['dimension']:11s} {r['bucket'][:34]:34s} high={r['highconf_accuracy']:.4f} agree={r['agree_accuracy']:.4f} gain={r['agree_minus_highconf_pp']:+.2f}pp")
    print('\nSaved:',OUT)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
