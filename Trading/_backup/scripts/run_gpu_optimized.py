"""
GPU-Optimized AlphaTrade - Default Configuration
================================================

NOTE: GPU-optimized settings are now the DEFAULT in Config class:
- n_envs: 64 (4x more parallel environments)
- batch_size: 4096 (8x larger GPU batches)
- mcts_batch_size: 128 (4x more MCTS roots)

This script is now just a convenience wrapper.
You can also run alphago_layering.py directly!

Usage:
    python run_gpu_optimized.py
    python run_gpu_optimized.py --symbols AAPL MSFT GOOGL --invert-trend

    # Or just use the main script (same defaults now):
    python alphago_layering.py --symbols AAPL MSFT --invert-trend
"""

import sys

# Import and run main (uses optimized defaults automatically)
from alphago_layering import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU-OPTIMIZED CONFIGURATION (NOW DEFAULT)")
    print("="*70)
    print("✅ n_envs:          64 (4x more parallel)")
    print("✅ batch_size:      4096 (8x larger GPU batches)")
    print("✅ mcts_batch_size: 128 (4x more MCTS roots)")
    print("✅ Parallel MCTS:   AUTO-ENABLED")
    print("")
    print("These are now the DEFAULT settings in Config class.")
    print("No special flags needed!")
    print("="*70)
    print("")

    main()
