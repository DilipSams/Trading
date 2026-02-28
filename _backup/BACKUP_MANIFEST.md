# Backup Manifest

**Created**: February 16, 2026
**Purpose**: Archive investigation, testing, and archived files from main codebase

---

## Folder Structure

```
_backup/
├── investigation_scripts/   # Test and investigation Python scripts
├── logs/                     # Log files from investigations
├── archived_versions/        # Old/deprecated Python files
└── run_outputs/             # Reserved for run output folders
```

---

## Files Moved

### Investigation Scripts (12 files)
- compare_configurations.py
- comprehensive_ic_test.py
- quick_ic_check.py
- quick_ic_test.py
- run_investigations.bat
- run_investigations.sh
- run_optimized.py
- test_integrity.py
- test_inversion_quick.py
- test_inversions.py
- test_optimized.py
- test_rl_observation_fix.py

### Logs (4 files)
- full_multi_horizon_test.log
- investigation_01_inversions.log
- investigation_01_inversions_v2.log
- test_inverted.log

### Archived Versions (2 files)
- alphago_architecture_inverted.py (old inverted version)
- alphago_system.py (deprecated system file)

---

## Main Directory (Kept)

### Core Python Files (9 files)
- alphago_architecture.py
- alphago_cost_model.py
- alphago_enhancements.py
- alphago_layering.py
- alphago_trading_system.py
- backtest_report.py
- data_quality.py
- run_artifacts.py
- validation_engine.py

### Documentation Files (12 .md files)
- alpha_research.md
- alpha_trade_study_guide.md
- CLAUDE.md
- curriculum_learning_fix.md
- embargo_analysis.md
- fixes_feb_16_2026.md
- INVESTIGATION_PLAN.md
- INVESTIGATIONS_README.md
- OPTIMIZED_CONFIG.md
- RL_zero_forensic.md
- self_play_research_findings.md
- TEMP_FILES_MANIFEST.md

---

## Restoration

If you need to restore any file:

```bash
# Example: Restore a test script
cp _backup/investigation_scripts/quick_ic_test.py .

# Example: Restore a log file
cp _backup/logs/investigation_01_inversions.log .
```

---

## Notes

- All .md documentation files kept in main directory (per user request)
- alphago_enhancements.py kept in main directory (per user request)
- Run output folders remain in run_output/ directory (not moved to backup)
- This backup structure is generic (not dated) for easy future additions
