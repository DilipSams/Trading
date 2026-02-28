"""One-shot script: run StockSelector and print top-100 symbol list."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '.')

from alphago_trading_system import (
    load_from_norgate, prepare_datasets, DEFAULT_SYMBOLS, SECTOR_MAP, Config,
    NORGATE_DIR,
)
from alphago_architecture import SelectionConfig, StockSelector

if __name__ == "__main__":
    NORGATE_PATH = r"D:\Experiments\norgate_data"
    cfg = Config()
    cfg.train_ratio = 0.8
    cfg.val_ratio   = 0.1

    print("Loading Norgate data...")
    raw = load_from_norgate(norgate_dir=NORGATE_PATH, symbols=DEFAULT_SYMBOLS)
    datasets = prepare_datasets(raw, cfg)
    print(f"Loaded {len(datasets)} datasets")

    # Build SPY returns lookup (same as alphago_layering.py)
    spy_returns_lookup = {}
    spy_path = Path(NORGATE_PATH) / "US_Equities" / "SPY.parquet"
    if spy_path.exists():
        import pandas as pd
        spy_df = pd.read_parquet(spy_path)
        col_map = {c: "Close" for c in spy_df.columns
                   if c.lower().strip() == "close" and "unadj" not in c.lower()}
        spy_df = spy_df.rename(columns=col_map)
        if "Close" in spy_df.columns:
            spy_closes = spy_df["Close"].values.astype(np.float64)
            spy_dates  = np.array(spy_df.index.astype(str))
            spy_rets   = np.diff(spy_closes) / (spy_closes[:-1] + 1e-12)
            for i, r in enumerate(spy_rets):
                spy_returns_lookup[spy_dates[i + 1][:10]] = float(r)
            print(f"SPY benchmark: {len(spy_returns_lookup)} daily returns")

    sel_cfg  = SelectionConfig(top_n=100)
    selector = StockSelector(sel_cfg, SECTOR_MAP)
    selected = selector.select(datasets, spy_returns_lookup)

    syms = [d.symbol for d in selected]
    log  = selector.selection_log[-1]

    print(f"\nTop 100 stocks by v8.0 composite score:")
    print(f"{'Rank':<5} {'Symbol':<7} {'Score':>8}  {'Momentum':>9}  {'SMA':>5}  {'RS/SPY':>8}  {'Vol':>6}")
    print("-" * 62)
    for rank, (sym, score, comp) in enumerate(log['rankings'][:100], 1):
        print(f"{rank:<5} {sym:<7} {score:>+8.4f}  "
              f"{comp['momentum']:>+8.1%}  "
              f"{comp['sma_score']:>4}/3  "
              f"{comp['rs_vs_spy']:>+7.1%}  "
              f"{comp['vol_20']:>5.0%}")

    print("\n--- SYMBOL LIST ---")
    print(",".join(syms))
