@echo off
REM Quick investigation runner for Windows
REM Tests the 4 key hypotheses from multi-horizon profiling

echo ========================================================================
echo              MULTI-HORIZON ALPHA INVESTIGATIONS
echo.
echo   Running 4 critical tests to validate signal inversions
echo   Estimated time: 8-10 hours total (2-2.5 hours each)
echo ========================================================================
echo.

REM Test 1: Signal Inversions
echo -----------------------------------------------------------------------
echo TEST 1: Signal Inversions (Seasonality + Vol Premium)
echo   Hypothesis: IC flips from negative to positive
echo   Started: %date% %time%
echo -----------------------------------------------------------------------
python alphago_layering.py --iterations 8 --steps-per-iter 80000 --invert-seasonality --invert-vol-premium > investigation_01_inversions.log 2>&1

echo [OK] Test 1 completed: %date% %time%
echo.

REM Test 2: Robust Ensemble
echo -----------------------------------------------------------------------
echo TEST 2: Robust Ensemble (Disable Seasonality + Vol Premium)
echo   Hypothesis: Sharpe improves by ^>20%%
echo   Started: %date% %time%
echo -----------------------------------------------------------------------
python alphago_layering.py --iterations 8 --steps-per-iter 80000 --disable-alphas "seasonality,vol_premium" > investigation_02_robust_ensemble.log 2>&1

echo [OK] Test 2 completed: %date% %time%
echo.

REM Test 3: Value Only
echo -----------------------------------------------------------------------
echo TEST 3: Value Alpha Isolation
echo   Hypothesis: Value alone achieves Sharpe ^> 0.5
echo   Started: %date% %time%
echo -----------------------------------------------------------------------
python alphago_layering.py --iterations 8 --steps-per-iter 80000 --disable-alphas "rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium" > investigation_03_value_only.log 2>&1

echo [OK] Test 3 completed: %date% %time%
echo.

REM Test 4: Optimal Configuration
echo -----------------------------------------------------------------------
echo TEST 4: Optimal Configuration
echo   Hypothesis: Combined inversions + disable RL/carry = best Sharpe
echo   Started: %date% %time%
echo -----------------------------------------------------------------------
python alphago_layering.py --iterations 8 --steps-per-iter 80000 --invert-seasonality --invert-vol-premium --disable-alphas "rl_ppo_mcts,carry" > investigation_04_optimal.log 2>&1

echo [OK] Test 4 completed: %date% %time%
echo.

REM Generate summary
echo ========================================================================
echo                      ALL TESTS COMPLETED
echo ========================================================================
echo.
echo Extracting results...

echo === MULTI-HORIZON IC COMPARISON === > investigation_summary.txt
echo. >> investigation_summary.txt

echo BASELINE (from full_multi_horizon_test.log): >> investigation_summary.txt
findstr /C:"MULTI-HORIZON ALPHA PROFILING" full_multi_horizon_test.log >> investigation_summary.txt
echo. >> investigation_summary.txt

echo TEST 1 (Inversions): >> investigation_summary.txt
findstr /C:"seasonality" investigation_01_inversions.log | findstr /C:"IC" >> investigation_summary.txt
findstr /C:"vol_premium" investigation_01_inversions.log | findstr /C:"IC" >> investigation_summary.txt
echo. >> investigation_summary.txt

echo TEST 2 (Robust Ensemble): >> investigation_summary.txt
findstr /C:"value" investigation_02_robust_ensemble.log | findstr /C:"IC" >> investigation_summary.txt
echo. >> investigation_summary.txt

echo TEST 3 (Value Only): >> investigation_summary.txt
findstr /C:"value" investigation_03_value_only.log | findstr /C:"IC" >> investigation_summary.txt
echo. >> investigation_summary.txt

echo === SHARPE RATIO COMPARISON === >> investigation_summary.txt
echo. >> investigation_summary.txt
findstr /C:"Sharpe:" full_multi_horizon_test.log >> investigation_summary.txt
findstr /C:"Sharpe:" investigation_01_inversions.log >> investigation_summary.txt
findstr /C:"Sharpe:" investigation_02_robust_ensemble.log >> investigation_summary.txt
findstr /C:"Sharpe:" investigation_03_value_only.log >> investigation_summary.txt
findstr /C:"Sharpe:" investigation_04_optimal.log >> investigation_summary.txt

echo.
echo [OK] Summary saved to: investigation_summary.txt
echo.
type investigation_summary.txt

pause
