#!/usr/bin/env python3
"""
AlphaGo Leveraged Rotation (ALR) — LETF Rotation Backtest.

6 strategies ranging from SPY buy-and-hold to a full long/short 3x LETF rotation
system using multi-index momentum, SMA trend filters, and volatility gating.

Importable as a module (call run_all_strategies()) or run standalone for the full report.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.environ.get("NORGATE_ROOT", r"D:\Experiments\norgate_data"), "US_Equities")
COST_BPS = 0.0005

TICKERS = ['UPRO', 'SPXU', 'TQQQ', 'SQQQ', 'UDOW', 'SDOW', 'SPY', 'QQQ', 'DIA']

STRATEGY_LABELS = {
    'S0': 'SPY B&H',
    'S1': 'TQQQ B&H',
    'S2': 'TQQQ/Cash (QQQ>200)',
    'S3': '3-Way Long Rotation',
    'S4': 'Long/Short Rotation',
    'S5': 'Full ALR (v9 LETF)',
}


def _sma(series, n):
    return series.rolling(n, min_periods=n).mean()


def _realized_vol(r, n=20):
    return r.rolling(n, min_periods=n).std() * np.sqrt(252)


def load_letf_data(data_dir=DATA_DIR):
    """Load LETF price data from Norgate parquets.

    Returns:
        pd.DataFrame with columns for each ticker, dropna-aligned, starting 2010-02-11.
        Returns None if any required ticker is missing.
    """
    frames = []
    for t in TICKERS:
        p = os.path.join(data_dir, t + '.parquet')
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)[['Close']].rename(columns={'Close': t})
        frames.append(df)
    data = pd.concat(frames, axis=1).dropna()
    data = data[data.index >= '2010-02-11']
    return data


def run_strategy(signals, prices, label, cost=COST_BPS):
    """Run a single strategy from signal series to equity curve."""
    equity = 1.0
    prev = 'CASH'
    eq = []
    dtl = []
    for dt, pos in signals.items():
        if dt not in prices.index:
            continue
        tc = cost if pos != prev else 0.0
        if pos == 'CASH':
            ret = 0.0
        else:
            idx = prices.index.get_loc(dt)
            if idx == 0:
                ret = 0.0
            else:
                ret = (prices[pos].iloc[idx] / prices[pos].iloc[idx - 1]) - 1.0
        equity *= (1 + ret - tc)
        prev = pos
        eq.append(equity)
        dtl.append(dt)
    return pd.Series(eq, index=dtl, name=label)


def compute_metrics(eq_curve, label):
    """Compute strategy metrics from an equity curve."""
    rets = eq_curve.pct_change().dropna()
    ny = len(eq_curve) / 252
    cagr = (eq_curve.iloc[-1] ** (1 / ny)) - 1
    sh = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0]
    sortino = (rets.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    ann_vol = rets.std() * np.sqrt(252) if len(rets) > 0 else 0
    rm = eq_curve.cummax()
    dd = (eq_curve - rm) / rm
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    ann = eq_curve.resample('YE').last().pct_change().dropna() * 100
    if len(ann) > 0:
        ann.iloc[0] = (eq_curve.resample('YE').last().iloc[0] - 1.0) * 100
    pp = (ann > 0).sum() / len(ann) * 100 if len(ann) > 0 else 0
    # Count signal changes for turnover
    return {
        'label': label,
        'cagr': cagr,         # as fraction (0.25 = 25%)
        'sharpe': sh,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': mdd,        # as fraction (-0.35 = -35%)
        'ann_vol': ann_vol,
        'pct_win_yrs': pp,
        'ann': ann,
        'n_years': ny,
        'period_start': str(eq_curve.index[0].date()) if len(eq_curve) > 0 else 'N/A',
        'period_end': str(eq_curve.index[-1].date()) if len(eq_curve) > 0 else 'N/A',
    }


def build_signals(close):
    """Build all 6 strategy signal series from OHLCV data.

    Args:
        close: DataFrame with columns SPY, QQQ, DIA, UPRO, SPXU, TQQQ, SQQQ, UDOW, SDOW

    Returns:
        dict[str, pd.Series] mapping strategy key (S0-S5) -> position signal series
    """
    spy_sma200 = _sma(close['SPY'], 200)
    spy_sma100 = _sma(close['SPY'], 100)
    qqq_sma200 = _sma(close['QQQ'], 200)
    qqq_sma100 = _sma(close['QQQ'], 100)
    dia_sma200 = _sma(close['DIA'], 200)
    dia_sma100 = _sma(close['DIA'], 100)
    mom_spy = close['SPY'] / close['SPY'].shift(21) - 1
    mom_qqq = close['QQQ'] / close['QQQ'].shift(21) - 1
    mom_dia = close['DIA'] / close['DIA'].shift(21) - 1
    rv = _realized_vol
    vol_upro = rv(close['UPRO'].pct_change())
    vol_tqqq = rv(close['TQQQ'].pct_change())
    vol_udow = rv(close['UDOW'].pct_change())
    vol_spxu = rv(close['SPXU'].pct_change())
    vol_sqqq = rv(close['SQQQ'].pct_change())
    vol_sdow = rv(close['SDOW'].pct_change())
    IM = {
        'SP': ('UPRO', 'SPXU', vol_upro, vol_spxu),
        'NDQ': ('TQQQ', 'SQQQ', vol_tqqq, vol_sqqq),
        'DOW': ('UDOW', 'SDOW', vol_udow, vol_sdow),
    }

    vd = close.index[close.index >= close.dropna().index[0]]
    signals = {k: pd.Series('CASH', index=vd) for k in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']}
    signals['S0'][::] = 'SPY'
    signals['S1'][::] = 'TQQQ'
    ph4, ph5 = 0, 0
    MH = 5
    pp4, pp5 = 'CASH', 'CASH'

    for i, dt in enumerate(vd):
        if dt not in close.index:
            continue
        if pd.isna(spy_sma200[dt]) or pd.isna(mom_spy[dt]):
            continue
        msp = float(mom_spy[dt])
        mndq = float(mom_qqq[dt]) if not pd.isna(mom_qqq[dt]) else -999
        mdow = float(mom_dia[dt]) if not pd.isna(mom_dia[dt]) else -999
        s200 = close['SPY'][dt] > spy_sma200[dt]
        n200 = close['QQQ'][dt] > qqq_sma200[dt] if not pd.isna(qqq_sma200[dt]) else False
        d200 = close['DIA'][dt] > dia_sma200[dt] if not pd.isna(dia_sma200[dt]) else False
        s100 = close['SPY'][dt] > spy_sma100[dt] if not pd.isna(spy_sma100[dt]) else False
        n100 = close['QQQ'][dt] > qqq_sma100[dt] if not pd.isna(qqq_sma100[dt]) else False
        d100 = close['DIA'][dt] > dia_sma100[dt] if not pd.isna(dia_sma100[dt]) else False
        mo = {'SP': msp, 'NDQ': mndq, 'DOW': mdow}
        u200 = {'SP': s200, 'NDQ': n200, 'DOW': d200}
        u100 = {'SP': s100, 'NDQ': n100, 'DOW': d100}
        signals['S2'][dt] = 'TQQQ' if n200 else 'CASH'
        if s200 or n200 or d200:
            el = {k: v for k, v in mo.items() if u200[k]}
            signals['S3'][dt] = IM[max(el, key=el.get)][0] if el else 'CASH'
        else:
            signals['S3'][dt] = 'CASH'
        ph4 = max(0, ph4 - 1)
        if ph4 > 0:
            signals['S4'][dt] = pp4
        else:
            if s200:
                el = {k: v for k, v in mo.items() if u200[k]}
                if el:
                    b = max(el, key=el.get)
                    etfl, _, vl, _ = IM[b]
                    v = float(vl[dt]) if not pd.isna(vl[dt]) else 0
                    ns4 = 'CASH' if v > 0.85 else etfl
                else:
                    ns4 = 'CASH'
            else:
                w = min(mo, key=mo.get)
                _, etfs, _, vs = IM[w]
                v = float(vs[dt]) if not pd.isna(vs[dt]) else 0
                ns4 = 'CASH' if v > 0.85 else etfs
            if ns4 != pp4:
                ph4 = MH
            signals['S4'][dt] = ns4
            pp4 = ns4
        ph5 = max(0, ph5 - 1)
        if ph5 > 0:
            signals['S5'][dt] = pp5
        else:
            nt = sum([s100, n100, d100])
            VC = 0.80
            if nt >= 2:
                el = {k: v for k, v in mo.items() if u100[k]}
                b = max(el, key=el.get)
                etfl, _, vl, _ = IM[b]
                v = float(vl[dt]) if not pd.isna(vl[dt]) else 0
                ns5 = 'CASH' if v > VC else etfl
            elif nt == 1:
                el = {k: v for k, v in mo.items() if u100[k]}
                b = max(el, key=el.get)
                etfl, _, vl, _ = IM[b]
                v = float(vl[dt]) if not pd.isna(vl[dt]) else 0
                if u200.get(b, False):
                    ns5 = 'CASH' if v > VC else etfl
                else:
                    ns5 = 'CASH'
            else:
                w = min(mo, key=mo.get)
                _, etfs, _, vs = IM[w]
                v = float(vs[dt]) if not pd.isna(vs[dt]) else 0
                ns5 = 'CASH' if v > VC else etfs
            if ns5 != pp5:
                ph5 = MH
            signals['S5'][dt] = ns5
            pp5 = ns5

    return signals


def run_all_strategies(data_dir=DATA_DIR, cost_bps=COST_BPS):
    """Run all 6 LETF rotation strategies and return equity curves + metrics.

    Returns:
        dict[str, dict] with keys per strategy:
            'equity': pd.Series (NAV starting at 1.0)
            'metrics': dict (cagr, sharpe, sortino, calmar, max_dd, ann_vol, ...)
            'signals': pd.Series (daily position labels)
        Returns None if LETF data is unavailable.
    """
    data = load_letf_data(data_dir)
    if data is None:
        return None

    close = data.copy()
    signals = build_signals(close)

    results = {}
    for key, sig in signals.items():
        label = STRATEGY_LABELS[key]
        eq = run_strategy(sig, close, label, cost_bps)
        m = compute_metrics(eq, label)
        # Count signal changes for turnover estimate
        trade_count = int((sig != sig.shift()).sum())
        m['trade_count'] = trade_count
        m['turnover_annual'] = trade_count / max(m['n_years'], 0.01)
        results[key] = {'equity': eq, 'metrics': m, 'signals': sig}

    return results


# ---------------------------------------------------------------------------
# Standalone report (when run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_letf_data()
    if data is None:
        print("ERROR: LETF data not found. Check Norgate path.")
        raise SystemExit(1)

    print(f'Data: {data.index[0].date()} to {data.index[-1].date()}  ({len(data)} trading days)')

    results = run_all_strategies()

    SEP = '=' * 100
    print('\n' + SEP)
    print('STRATEGY COMPARISON \u2014 AlphaGo Leveraged Rotation (ALR)')
    print(f'Period: {data.index[0].date()} \u2014 {data.index[-1].date()}   |   Costs: {int(COST_BPS*10000)}bps/trade one-way')
    print(SEP)
    print('%-28s  %7s  %7s  %8s  %7s  %7s  %s' % ('Strategy', 'CAGR', 'Sharpe', 'MaxDD', 'Calmar', 'WinYrs', '$10k->'))
    print('-' * 100)
    for key in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']:
        eq = results[key]['equity']
        m = results[key]['metrics']
        final = eq.iloc[-1] * 10_000
        row = '  %-26s  %6.1f%%  %7.2f  %7.1f%%  %7.2f  %6.0f%%' % (
            m['label'], m['cagr'] * 100, m['sharpe'], m['max_dd'] * 100, m['calmar'], m['pct_win_yrs'])
        row += '  $%-10s' % '{:,.0f}'.format(final)
        print(row)

    print('\n' + SEP)
    print('YEAR-BY-YEAR RETURNS (%)')
    print(SEP)
    all_years = sorted(set().union(*[set(results[k]['metrics']['ann'].index.year) for k in results]))
    sk = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    ss = ['SPY', 'TQQQ B&H', 'TQQQ/Cash', '3-Way-L', 'L/S-Rot', 'ALR-Full']
    print('  %-6s' % 'Year' + ''.join('  %10s' % x for x in ss))
    print('  ' + '-' * 6 + ('  ' + '-' * 10) * len(sk))
    for yr in all_years:
        row = '  %-6s' % yr
        for key in sk:
            ann = results[key]['metrics']['ann']
            ya = ann[ann.index.year == yr]
            if len(ya) > 0:
                row += '  %+9.1f%%' % ya.iloc[0]
            else:
                row += '  %10s' % 'N/A'
        print(row)
    print('  ' + '-' * 6 + ('  ' + '-' * 10) * len(sk))
    for rname in ['Total', 'CAGR/yr']:
        row = '  %-6s' % rname
        for key in sk:
            ann = results[key]['metrics']['ann']
            vals = ann.values / 100
            comp = 1.0
            for v in vals:
                comp *= (1 + v)
            if rname == 'Total':
                res = (comp - 1.0) * 100
            else:
                res = ((comp ** (1.0 / len(vals))) - 1.0) * 100 if len(vals) > 0 else 0
            row += '  %+9.1f%%' % res
        print(row)

    print('\n  TRADE COUNTS (signal changes):')
    for key in sk[2:]:
        tc = results[key]['metrics']['trade_count']
        lbl = results[key]['metrics']['label']
        print(f'    {lbl}: {tc} signal changes')

    print(f'\n  NOTE: Costs = {int(COST_BPS*10000)}bps per ONE-WAY trade (buy or sell).')
    print('  TQQQ/UPRO/UDOW: launched Feb 2010. Backtest period: ~16 years.')
    print('  Cash = 0% return (no T-bill credit). This understates real returns slightly.')
    print(SEP)
