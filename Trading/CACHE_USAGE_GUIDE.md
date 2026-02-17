# Data Caching System - Usage Guide

## Overview

The trading system now includes intelligent data caching to avoid re-downloading market data on every run. This saves time and reduces API calls to yfinance.

## Features

1. **Automatic Caching**: Downloaded data is automatically saved to `data_cache/` directory
2. **Freshness Check**: Cache expires after 24 hours (configurable)
3. **Interactive Prompts**: System asks before using stale cache or re-downloading
4. **Metadata Tracking**: Tracks when data was last downloaded

## Default Behavior

When you run a backtest:

```bash
python alphago_layering.py --invert-trend --iterations 3
```

**First Run**:
- No cache found → downloads fresh data
- Saves to `data_cache/market_data_*.pkl`
- Creates timestamp file `*_meta.txt`

**Subsequent Runs (within 24 hours)**:
```
Found cached data from 2026-02-16 10:30:15 (2.5h ago)

  Use cached data? (Y/n):
```

- Press `Enter` or `Y` → uses cache (instant load)
- Type `n` → re-downloads fresh data

**After 24 Hours**:
```
Cache is 26.3h old (>24.0h). Re-download? (Y/n):
```

- Press `Enter` or `Y` → re-downloads fresh data
- Type `n` → uses stale cache anyway (user override)

## Command-Line Options

### 1. Change Cache Location

```bash
python alphago_layering.py --cache-dir "my_custom_cache"
```

### 2. Change Cache Expiry (in hours)

```bash
# 6 hour expiry (for intraday trading)
python alphago_layering.py --cache-max-age 6

# 7 day expiry (for long-term backtests)
python alphago_layering.py --cache-max-age 168
```

### 3. Force Re-Download (Skip Cache)

```bash
# Always download fresh data without prompting
python alphago_layering.py --force-download
```

### 4. Disable Caching Entirely

```bash
# No caching, always download
python alphago_layering.py --no-cache
```

## Use Cases

### Intraday Trading (Fresh Data Needed)

```bash
python alphago_layering.py \
    --timeframes "5m,15m,30m,1h" \
    --cache-max-age 1 \
    --force-download
```

### Daily Backtesting (Cache OK)

```bash
python alphago_layering.py \
    --timeframes "1d" \
    --cache-max-age 24
```

### Historical Research (No Fresh Data Needed)

```bash
python alphago_layering.py \
    --timeframes "1d" \
    --cache-max-age 168  # 7 days
```

### Quick Iteration (Always Use Cache)

```bash
# Download once
python alphago_layering.py --iterations 1

# Then iterate with cache (instant startup)
python alphago_layering.py --iterations 3
# (Will prompt, press Y to use cache)
```

## Cache Files

The cache system creates:

```
data_cache/
├── market_data_AAPL_AMD_DIA_53syms_5m_15m_30m_1h_1d.pkl  # Actual data
└── market_data_AAPL_AMD_DIA_53syms_5m_15m_30m_1h_1d_meta.txt  # Timestamp
```

**File naming**:
- First 3 symbols (alphabetically): `AAPL_AMD_DIA`
- Total symbol count: `53syms`
- Timeframes: `5m_15m_30m_1h_1d`

## Cache Invalidation

Cache is automatically invalidated when:

1. **Time expires**: Older than `--cache-max-age` hours
2. **Symbol list changes**: Different symbols = different cache file
3. **Timeframes change**: Different TFs = different cache file
4. **Manual override**: User types `n` when prompted

## Managing Cache

### View Cache Size

```bash
du -sh data_cache/
```

### Clear Old Cache

```bash
# Remove all cached data
rm -rf data_cache/

# Remove caches older than 7 days (Unix)
find data_cache/ -name "*.pkl" -mtime +7 -delete
```

### Selective Cache Clear

```bash
# Keep daily data, remove intraday
rm data_cache/*5m*.pkl
rm data_cache/*15m*.pkl
rm data_cache/*30m*.pkl
rm data_cache/*1h*.pkl
```

## Performance Impact

### Without Caching

- **50 symbols × 1d**: ~10-15 seconds download
- **50 symbols × 5 TFs**: ~2-5 minutes download
- **53 symbols × 5 TFs**: ~3-6 minutes download

### With Caching (Cache Hit)

- **Any configuration**: <1 second load
- **Memory**: ~500MB for 53 symbols × 5 TFs

### Space Usage

- **Daily data only**: ~10-20 MB/cache
- **5 timeframes**: ~100-200 MB/cache
- **Intraday 5m (5 years)**: ~500MB-1GB/cache

## Best Practices

1. **Set expiry based on use case**:
   - Live/paper trading: 1-6 hours
   - Daily backtests: 24 hours
   - Historical research: 7 days (168 hours)

2. **Use `--force-download` when**:
   - Testing new market data
   - After market close (to get today's data)
   - Debugging data issues

3. **Clean cache periodically**:
   - Before major backtests (ensure fresh data)
   - When low on disk space
   - After changing symbol universe

4. **Disable cache when**:
   - Running on CI/CD (always fresh)
   - Testing data download logic
   - Using multiple machines (shared cache not supported)

## Troubleshooting

### "Cache metadata error" Warning

**Cause**: Corrupted timestamp file

**Fix**:
```bash
rm data_cache/*_meta.txt
python alphago_layering.py --force-download
```

### Cache Not Working

**Check**:
1. Permissions: Can write to `data_cache/`?
2. Space: Enough disk space?
3. Path: Is `--cache-dir` correct?

**Debug**:
```bash
python alphago_layering.py --force-download --cache-dir "debug_cache"
ls -lh debug_cache/
```

### Always Prompting (Never Using Cache)

**Likely**: Symbol list or timeframes changing between runs

**Check**:
```bash
# See what cache files exist
ls data_cache/*.pkl

# Compare to current run parameters
python alphago_layering.py --n-symbols 53 --timeframes "1d"
```

## Integration with Existing Workflows

### Automated Backtests (Cron/Scheduled)

```bash
#!/bin/bash
# Daily backtest with fresh data

python alphago_layering.py \
    --force-download \
    --iterations 5 \
    --steps-per-iter 40000 \
    --cache-max-age 24
```

### Development/Testing (Fast Iteration)

```bash
# Download once at start of day
python alphago_layering.py --iterations 1 --force-download

# Then iterate quickly with cache
for i in {1..10}; do
    python alphago_layering.py --iterations 3 <<< "y"  # Auto-answer Y
done
```

### Multi-Configuration Tests

```bash
# Download once, then test different configs
python alphago_layering.py --force-download --iterations 1

# Test config 1 (uses cache)
python alphago_layering.py --invert-trend --iterations 3 <<< "y"

# Test config 2 (uses same cache)
python alphago_layering.py --invert-vol-premium --iterations 3 <<< "y"
```

## Advanced: Programmatic Usage

If calling from Python scripts:

```python
from alphago_trading_system import download_data
from alphago_architecture import ArchitectureConfig

cfg = ArchitectureConfig()
symbols = ["AAPL", "MSFT", "GOOGL"]

# With caching (default 24h)
data = download_data(symbols, cfg, cache_dir="data_cache", cache_max_age_hours=24)

# Force fresh download
data = download_data(symbols, cfg, force_download=True)

# Disable cache (old behavior)
data = download_data(symbols, cfg, cache_max_age_hours=0)
```

## FAQ

**Q: Does cache work across different experiments?**
A: Yes! Same symbols + timeframes = same cache file, regardless of other parameters.

**Q: Can I pre-download data?**
A: Yes! Run once with `--force-download`, then all future runs use cache.

**Q: Does this work with `--data-dir`?**
A: No, caching only applies to yfinance downloads. `--data-dir` loads directly from files.

**Q: What if I want longer history than yfinance allows?**
A: Intraday data is limited by yfinance (5m = 60 days max). Use daily for longer history.

**Q: Can I share cache between machines?**
A: Not officially supported, but you can manually copy `data_cache/` directory.

---

**Last Updated**: 2026-02-16
**Cache Version**: 1.0
**Compatible With**: Alpha-Trade v7.0+
