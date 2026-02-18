# Temporary Testing & Analysis Files

**Created**: 2026-02-15
**Purpose**: Track temporary files created for IC analysis and optimization testing

---

## Files Created (Not Part of Core Codebase)

### Testing & Validation Scripts

1. **quick_ic_test.py**
   - Purpose: Fast IC validation on 5 symbols (3 seconds)
   - Status: Keep - useful for quick validation
   - Usage: `python quick_ic_test.py --invert-trend`

2. **compare_configurations.py**
   - Purpose: Compare multiple alpha configurations side-by-side
   - Status: Keep - useful for A/B testing configurations
   - Usage: `python compare_configurations.py`

3. **test_optimized.py**
   - Purpose: Compare baseline vs optimized configuration
   - Status: Archive - validates 175% improvement (one-time test)
   - Usage: `python test_optimized.py`

4. **test_inversions.py** (if exists)
   - Purpose: Test signal inversions for seasonality and vol_premium
   - Status: Archive - early investigation script
   - Usage: Obsolete - use quick_ic_test.py instead

5. **test_inversion_quick.py** (if exists)
   - Purpose: Quick signal inversion testing
   - Status: Archive - superseded by quick_ic_test.py
   - Usage: Obsolete

6. **test_integrity.py** ⚠️ NOT TEMP - ORIGINAL CORE FILE
   - Purpose: Core data integrity validation (1,017 lines)
   - Status: Core file - DO NOT MOVE OR DELETE
   - Note: This file was in the original codebase

7. **run_optimized.py**
   - Purpose: Deployment helper script for optimized config
   - Status: Keep - simplifies production deployment
   - Usage: `python run_optimized.py --production`

### Documentation Files

5. **OPTIMIZED_CONFIG.md**
   - Purpose: Complete documentation of optimized configuration
   - Status: Keep - critical reference document
   - Contains: Performance metrics, implementation details, usage instructions

6. **INVESTIGATION_PLAN.md** (created earlier)
   - Purpose: Investigation plan for signal inversions and alpha validation
   - Status: Archive - historical reference
   - Contains: Original hypotheses and test plan

7. **TEMP_FILES_MANIFEST.md** (this file)
   - Purpose: Track all temporary files
   - Status: Keep - file management reference

### Output Directories (Not Tracked in Git)

8. **run_output/** - Various test run outputs
   - Pipeline v6.0_* folders
   - Pipeline v7.0_* folders
   - Status: Can delete after validation complete

9. **runs/** - Run manifests and step logs
   - Pipeline v6.0_* folders
   - Status: Can delete after validation complete

---

## Core Codebase Files (Modified, Keep)

1. **alphago_architecture.py**
   - Changes: Added trend inversion support, updated horizons to 15 bars
   - Lines modified: 251-290 (horizons), 666-705 (TrendAlpha inversion), 4631-4668 (pipeline builder)
   - Status: Core file - keep all changes

2. **alphago_layering.py** (assumed modified based on docs)
   - Changes: Added --invert-trend CLI flag
   - Status: Core file - keep all changes

---

## Cleanup Recommendations

### Can Delete After Production Deployment
- `run_output/Pipeline v6.0_*` - Old test runs
- `run_output/Pipeline v7.0_*` - Old test runs (except latest)
- `runs/Pipeline v6.0_*` - Old run manifests

### Archive (Move to /tests/archive)
- `test_optimized.py` - One-time baseline vs optimized validation
- `test_inversions.py` - Early signal inversion tests (obsolete)
- `test_inversion_quick.py` - Quick inversion tests (superseded)
- `INVESTIGATION_PLAN.md` - Historical investigation plan

### ⚠️ DO NOT MOVE - Original Core Files
- `test_integrity.py` - Original core file (1,017 lines) - data integrity validation

### Keep Permanently
- `quick_ic_test.py` - Fast validation tool (useful for ongoing testing)
- `compare_configurations.py` - Multi-config testing (useful for A/B tests)
- `run_optimized.py` - Deployment helper (simplifies production runs)
- `OPTIMIZED_CONFIG.md` - Key documentation
- `TEMP_FILES_MANIFEST.md` - This file (for reference)

---

## Cleanup Commands

```bash
# Archive old test runs
mkdir -p archive/test_runs
mv "run_output/Pipeline v6.0_"* archive/test_runs/
mv "runs/Pipeline v6.0_"* archive/test_runs/

# Archive obsolete test scripts (DO NOT MOVE test_integrity.py - it's a core file!)
mkdir -p tests/archive
mv test_optimized.py tests/archive/
mv test_inversions.py tests/archive/ 2>/dev/null || true  # May not exist
mv test_inversion_quick.py tests/archive/ 2>/dev/null || true  # May not exist

# Keep useful testing scripts in tests folder
mkdir -p tests
mv quick_ic_test.py tests/
mv compare_configurations.py tests/

# Move investigation docs to docs folder (optional)
mkdir -p docs
mv INVESTIGATION_PLAN.md docs/
mv OPTIMIZED_CONFIG.md docs/
```

---

## File Size Summary (Estimated)

| Category | Files | Est. Size |
|----------|-------|-----------|
| Testing scripts (keep) | 3 | ~15 KB |
| Testing scripts (archive) | 3 | ~10 KB |
| Documentation | 3 | ~30 KB |
| Test outputs | ~20 folders | ~5-10 MB |
| Core changes | 2 files | Modifications only |

**Total temporary footprint**: ~10-15 MB
**Archivable**: ~5-10 MB (test outputs + obsolete scripts)

---

**Last Updated**: 2026-02-15
**Review Date**: After production deployment validation
