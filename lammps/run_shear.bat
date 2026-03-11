@echo off
cd /d "%~dp0"

echo ========================================
echo   SLLOD Multi-Rate Shear Simulations
echo   Start: %date% %time%
echo ========================================

echo.
echo [1/5] gamma_dot=0.003  DT=0.003  Start: %time%
mpiexec -n 22 lmp -in in.shear_template -var SHEAR_RATE 0.003 -var DT 0.003 -var SEED 12345 -log log.shear_0.003
if errorlevel 1 ( echo [ERROR] gamma_dot=0.003 failed & pause & exit /b 1 )
echo [1/5] gamma_dot=0.003  Done: %time%

echo.
echo [2/5] gamma_dot=0.005  DT=0.003  Start: %time%
mpiexec -n 22 lmp -in in.shear_template -var SHEAR_RATE 0.005 -var DT 0.003 -var SEED 12345 -log log.shear_0.005
if errorlevel 1 ( echo [ERROR] gamma_dot=0.005 failed & pause & exit /b 1 )
echo [2/5] gamma_dot=0.005  Done: %time%

echo.
echo [3/5] gamma_dot=0.015  DT=0.001  Start: %time%
mpiexec -n 22 lmp -in in.shear_template -var SHEAR_RATE 0.015 -var DT 0.001 -var SEED 12345 -log log.shear_0.015
if errorlevel 1 ( echo [ERROR] gamma_dot=0.015 failed & pause & exit /b 1 )
echo [3/5] gamma_dot=0.015  Done: %time%

echo.
echo [4/5] gamma_dot=0.030  DT=0.001  Start: %time%
mpiexec -n 22 lmp -in in.shear_template -var SHEAR_RATE 0.030 -var DT 0.001 -var SEED 12345 -log log.shear_0.030
if errorlevel 1 ( echo [ERROR] gamma_dot=0.030 failed & pause & exit /b 1 )
echo [4/5] gamma_dot=0.030  Done: %time%

echo.
echo [5/5] gamma_dot=0.060  DT=0.001  Start: %time%
mpiexec -n 22 lmp -in in.shear_template -var SHEAR_RATE 0.060 -var DT 0.001 -var SEED 12345 -log log.shear_0.060
if errorlevel 1 ( echo [ERROR] gamma_dot=0.060 failed & pause & exit /b 1 )
echo [5/5] gamma_dot=0.060  Done: %time%

echo.
echo ========================================
echo   All done!  End: %date% %time%
echo ========================================
pause
