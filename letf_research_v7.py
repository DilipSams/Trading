import os, warnings, calendar
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.environ.get("NORGATE_ROOT", r"C:\ProgramData\NorgateData"), "US_Equities")
START_DATE = "2010-03-11"

SIGNAL_ETFS   = ["XLK", "XLE", "XLF", "QQQ", "IWM", "XLV"]
POSITION_ETFS = ["TECL", "ERX", "FAS", "TQQQ", "TNA", "CURE"]
SIGNAL_TO_POS = dict(zip(SIGNAL_ETFS, POSITION_ETFS))

TRANSACTION_COST = 0.0010
MONTHLY          = 21
WEEKLY           = 5
MOMENTUM_WEIGHTS = {21: 0.40, 63: 0.35, 126: 0.25}

def load_close(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df["Close"].sort_index().rename(ticker)

print("Loading data ...")
spy       = load_close("SPY")
signals   = {s: load_close(s) for s in SIGNAL_ETFS}
positions = {p: load_close(p) for p in POSITION_ETFS}

all_signal = pd.DataFrame(signals).loc[START_DATE:].dropna(how="all")
common_idx = all_signal.index
spy    = spy.reindex(common_idx).ffill()
pos_df = pd.DataFrame(positions).reindex(common_idx).ffill()
sig_df = all_signal.ffill()

print(f"Date range  : {common_idx[0].date()} -> {common_idx[-1].date()}")
print(f"Trading days: {len(common_idx)}")

def compute_momentum(prices):
    scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for window, weight in MOMENTUM_WEIGHTS.items():
        scores = scores + prices.pct_change(window) * weight
    return scores.shift(1)

mom_scores = compute_momentum(sig_df)
spy_sma200 = spy.rolling(200).mean().shift(1)
bear_gate  = spy.shift(1) < spy_sma200

def backtest2(rebalance_every=21, bear_gate_on=False, abs_mom_filter=False,
              top_n=1, min_hold=0):
    n         = len(common_idx)
    nav_vals  = np.ones(n)
    holding   = ["Cash"] * top_n
    days_held = 0
    switches  = []
    one_m_ret = sig_df.pct_change(21).shift(1)
    for i in range(1, n):
        date      = common_idx[i]
        prev_date = common_idx[i - 1]
        rebal     = (i % rebalance_every == 0)
        if min_hold > 0 and days_held < min_hold:
            rebal = False
        new_holding = holding[:]
        if rebal:
            scores = mom_scores.loc[date]
            if bear_gate_on and bear_gate.loc[date]:
                new_holding = ["Cash"] * top_n
            else:
                ranked     = scores.dropna().sort_values(ascending=False)
                candidates = list(ranked.index[:top_n])
                if abs_mom_filter:
                    om         = one_m_ret.loc[date]
                    candidates = [c for c in candidates if om.get(c, 0) > 0]
                if len(candidates) == 0:
                    new_holding = ["Cash"] * top_n
                elif len(candidates) < top_n:
                    new_holding  = [SIGNAL_TO_POS[c] for c in candidates]
                    new_holding += ["Cash"] * (top_n - len(candidates))
                else:
                    new_holding = [SIGNAL_TO_POS[c] for c in candidates[:top_n]]
        n_sw = sum(1 for a, b in zip(holding, new_holding) if a != b)
        if n_sw > 0:
            switches.append((date, new_holding[:]))
            holding   = new_holding[:]
            days_held = 0
        else:
            days_held += 1
        slot_rets = []
        for h in holding:
            if h == "Cash":
                slot_rets.append(0.0)
            elif h in pos_df.columns:
                pn = pos_df.loc[date, h]
                pp = pos_df.loc[prev_date, h]
                slot_rets.append(pn / pp - 1 if (not np.isnan(pn)) and pp > 0 else 0.0)
            else:
                slot_rets.append(0.0)
        daily_ret   = float(np.mean(slot_rets))
        cost_mult   = (1.0 - TRANSACTION_COST) ** n_sw
        nav_vals[i] = nav_vals[i-1] * (1.0 + daily_ret) * cost_mult
    return pd.Series(nav_vals, index=common_idx), switches

def bah_nav(ticker):
    if ticker in pos_df.columns:
        p = pos_df[ticker]
    elif ticker in sig_df.columns:
        p = sig_df[ticker]
    else:
        p = load_close(ticker).reindex(common_idx).ffill()
    p = p.dropna()
    return (p / p.iloc[0]).reindex(common_idx).ffill()

print("\nRunning strategies ...")
navs         = {}
all_switches = {}
navs["B0_TQQQ"] = bah_nav("TQQQ")
navs["B1_SPY"]  = bah_nav("SPY")
navs["B2_QQQ"]  = bah_nav("QQQ")
configs = [
    ("S1", dict(rebalance_every=MONTHLY, bear_gate_on=False, abs_mom_filter=False, top_n=1)),
    ("S2", dict(rebalance_every=MONTHLY, bear_gate_on=True,  abs_mom_filter=False, top_n=1)),
    ("S3", dict(rebalance_every=MONTHLY, bear_gate_on=False, abs_mom_filter=True,  top_n=1)),
    ("S4", dict(rebalance_every=MONTHLY, bear_gate_on=True,  abs_mom_filter=True,  top_n=1)),
    ("S5", dict(rebalance_every=MONTHLY, bear_gate_on=True,  abs_mom_filter=False, top_n=2)),
    ("S6", dict(rebalance_every=WEEKLY,  bear_gate_on=True,  abs_mom_filter=False, top_n=1)),
    ("S7", dict(rebalance_every=MONTHLY, bear_gate_on=True,  abs_mom_filter=False, top_n=1, min_hold=MONTHLY)),
]
for k, cfg in configs:
    print(f"  Running {k} ...")
    navs[k], all_switches[k] = backtest2(**cfg)
print("All strategies complete.")

def calc_metrics(nav_series):
    nav = nav_series.dropna()
    if len(nav) < 2:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"), final=float("nan"))
    n_years  = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr     = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1
    dr       = nav.pct_change().dropna()
    sharpe   = dr.mean() / dr.std() * 252**0.5 if dr.std() > 0 else float("nan")
    roll_max = nav.cummax()
    maxdd    = ((nav - roll_max) / roll_max).min()
    return dict(cagr=cagr, sharpe=sharpe, maxdd=maxdd, final=nav.iloc[-1])

strat_labels = {
    "B0_TQQQ": "B0  TQQQ B&H",
    "B1_SPY" : "B1  SPY  B&H",
    "B2_QQQ" : "B2  QQQ  B&H",
    "S1"     : "S1  Top-1 No Gate",
    "S2"     : "S2  Top-1 Bear Gate",
    "S3"     : "S3  Top-1 Abs Mom Filter",
    "S4"     : "S4  Top-1 Bear+AbsMom",
    "S5"     : "S5  Top-2 Bear Gate",
    "S6"     : "S6  Weekly Bear Gate",
    "S7"     : "S7  Top-1 Bear+MinHold21",
}
SEP = "=" * 80

def annual_returns(nav_series):
    nav = nav_series.dropna()
    ann = {}
    for yr in sorted(nav.index.year.unique()):
        yr_nav = nav[nav.index.year == yr]
        if len(yr_nav) >= 2:
            ann[yr] = yr_nav.iloc[-1] / yr_nav.iloc[0] - 1
    return ann

all_ann   = {k: annual_returns(v) for k, v in navs.items()}
all_years = sorted({yr for d in all_ann.values() for yr in d})

def dominant_holding_by_year(switches, nav_series):
    sw_dict  = {d: h for d, h in switches}
    current  = "Cash"
    hold_ser = {}
    for date in nav_series.dropna().index:
        if date in sw_dict:
            current = sw_dict[date][0]
        hold_ser[date] = current
    hs = pd.Series(hold_ser)
    return {yr: hs[hs.index.year == yr].value_counts().idxmax()
            for yr in hs.index.year.unique()}

def build_holding_series(strategy_key):
    sw_dict = {d: h for d, h in all_switches.get(strategy_key, [])}
    current = "Cash"
    result  = {}
    for date in common_idx:
        if date in sw_dict:
            current = sw_dict[date][0]
        result[date] = current
    return pd.Series(result)

def month_by_month_analysis(year, strategy_key):
    idx_yr = common_idx[common_idx.year == year]
    if len(idx_yr) == 0:
        print(f"  No data for {year}")
        return
    hold_ser = build_holding_series(strategy_key)
    months   = sorted(idx_yr.month.unique())
    print(f"\n  Strategy: {strategy_key}  |  Year: {year}")
    print(f"  {'Month':<8} {'TopSignal':<12} {'Held3x':<8} {'BearGate':<10} {'3x MthRet':>10} {'SPY MthRet':>11}")
    print("  " + "-" * 64)
    for mo in months:
        mo_dates   = idx_yr[idx_yr.month == mo]
        first_date = mo_dates[0]
        last_date  = mo_dates[-1]
        sc         = mom_scores.loc[first_date].dropna()
        top_signal = sc.idxmax() if len(sc) > 0 else "N/A"
        bg         = bool(bear_gate.loc[first_date])
        ah         = hold_ser.get(first_date, "Cash")
        if ah != "Cash" and ah in pos_df.columns:
            p0  = pos_df.loc[first_date, ah]
            p1  = pos_df.loc[last_date,  ah]
            mrt = p1 / p0 - 1 if p0 > 0 else 0.0
        else:
            mrt = 0.0
        spy_ret = spy.loc[last_date] / spy.loc[first_date] - 1
        mo_name = calendar.month_abbr[mo]
        bg_str  = "YES" if bg else "no"
        print(f"  {mo_name:<8} {top_signal:<12} {ah:<8} {bg_str:<10} {mrt:>10.1%} {spy_ret:>11.1%}")

print()
print(SEP)
print("SECTION 1 -- SUMMARY TABLE")
print(SEP)
print(f"{'Strategy':<28} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Switches':>9} {'Final':>10}")
print("-" * 80)
for key, label in strat_labels.items():
    m   = calc_metrics(navs[key])
    nsw = len(all_switches.get(key, []))
    cagr_s = f"{m['cagr']:>7.1%}"
    sha_s  = f"{m['sharpe']:>8.2f}"
    mdd_s  = f"{m['maxdd']:>7.1%}"
    fin_s  = f"{m['final']:>10.2f}"
    print(f"{label:<28} {cagr_s} {sha_s} {mdd_s} {nsw:>9} {fin_s}")

print()
print(SEP)
print("SECTION 2 -- YEAR-BY-YEAR RETURNS")
print(SEP)
col_w = 9
print(f"{'Year':<6}" + "".join(f"{k:>{col_w}}" for k in strat_labels))
print("-" * (6 + col_w * len(strat_labels)))
for yr in all_years:
    row = f"{yr:<6}"
    for k in strat_labels:
        val = all_ann[k].get(yr, float("nan"))
        row += f"{'N/A':>{col_w}}" if (val != val) else f"{val:>{col_w}.1%}"
    print(row)

print()
print(SEP)
print("SECTION 3 -- WHAT DID WE HOLD? (dominant 3x ETF per year)")
print(SEP)
hold_s1 = dominant_holding_by_year(all_switches.get("S1", []), navs["S1"])
hold_s2 = dominant_holding_by_year(all_switches.get("S2", []), navs["S2"])
hold_s4 = dominant_holding_by_year(all_switches.get("S4", []), navs["S4"])
print(f"{'Year':<6} {'S1 (No Gate)':<16} {'S2 (Bear Gate)':<16} {'S4 (Bear+AbsMom)':<18}")
print("-" * 60)
for yr in all_years:
    s1h = hold_s1.get(yr, "N/A")
    s2h = hold_s2.get(yr, "N/A")
    s4h = hold_s4.get(yr, "N/A")
    print(f"{yr:<6} {s1h:<16} {s2h:<16} {s4h:<18}")

print()
print(SEP)
print("SECTION 4 -- 2022 BEAR MARKET ANALYSIS (month-by-month)")
print(SEP)
month_by_month_analysis(2022, "S2")
month_by_month_analysis(2022, "S4")

print()
print(SEP)
print("SECTION 5 -- 2023 RECOVERY ANALYSIS (month-by-month)")
print(SEP)
month_by_month_analysis(2023, "S2")
month_by_month_analysis(2023, "S4")

print()
print(SEP)
print("SECTION 6 -- MOMENTUM SCORES AT START OF EACH MONTH (2022-2023)")
print(SEP)
for year in [2022, 2023]:
    idx_yr = common_idx[common_idx.year == year]
    months = sorted(idx_yr.month.unique())
    print(f"\n  Year {year}:")
    print(f"  {'Date':<12} " + "".join(f"{s:>8}" for s in SIGNAL_ETFS) + f"  {'Winner':<8} {'3xETF':<8}")
    print("  " + "-" * (12 + 8*len(SIGNAL_ETFS) + 20))
    for mo in months:
        first_date = idx_yr[idx_yr.month == mo][0]
        sc         = mom_scores.loc[first_date]
        valid      = sc.dropna()
        winner     = valid.idxmax() if len(valid) > 0 else "N/A"
        row = f"  {str(first_date.date()):<12} "
        for s in SIGNAL_ETFS:
            v = sc.get(s, float("nan"))
            row += f"{'N/A':>8}" if (v != v) else f"{v:>8.3f}"
        row += f"  {winner:<8} {SIGNAL_TO_POS.get(winner, 'N/A'):<8}"
        print(row)

print()
print(SEP)
print("SECTION 7 -- TOP DRAWDOWN PERIODS (S2 and S4)")
print(SEP)

def find_drawdowns(nav, top_n=5):
    roll_max = nav.cummax()
    results  = []
    in_dd    = False
    peak_date = nav.index[0]
    trough_date = trough_val = None
    for date, val in nav.items():
        if not in_dd:
            if val >= roll_max.loc[date] * 0.9999:
                peak_date = date
            else:
                in_dd = True
                peak_date   = nav[:date].idxmax()
                trough_date = date
                trough_val  = val
        else:
            if val < trough_val:
                trough_date = date
                trough_val  = val
            if val >= roll_max.loc[date] * 0.9999:
                dd_pct = trough_val / nav.loc[peak_date] - 1
                results.append((peak_date, trough_date, date, dd_pct))
                in_dd = False
                peak_date = date
    results.sort(key=lambda x: x[3])
    return results[:top_n]

for strat in ["S2", "S4"]:
    print(f"\n  {strat} -- top drawdowns:")
    print(f"  {'Peak':<12} {'Trough':<12} {'Recovery':<12} {'DD%':>8} {'Days to Trough':>15}")
    print("  " + "-" * 64)
    for peak, trough, rec, dd_pct in find_drawdowns(navs[strat].dropna()):
        print(f"  {str(peak.date()):<12} {str(trough.date()):<12} {str(rec.date()):<12} {dd_pct:>7.1%} {(trough-peak).days:>15}")

print()
print(SEP)
print("SECTION 8 -- FULL SWITCH HISTORY (S2 and S4)")
print(SEP)
for strat in ["S2", "S4"]:
    sw = all_switches.get(strat, [])
    print(f"\n  {strat}  ({len(sw)} total switches):")
    print(f"  {'Date':<12} New Holding")
    print("  " + "-" * 30)
    for d, h in sw:
        print(f"  {str(d.date()):<12} {', '.join(h)}")

print()
print(SEP)
print(f"COMPLETE. Evaluated {common_idx[0].date()} -> {common_idx[-1].date()}")
print(SEP)
