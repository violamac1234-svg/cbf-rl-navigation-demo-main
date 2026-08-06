@echo off
setlocal
set "PYTHON_EXE=D:\anaconda3\envs\cbf_learning\python.exe"

rem Stage 1: four base methods only; stop immediately if one fails.
for %%M in (nominal reward_only filter_only dual) do (
    "%PYTHON_EXE%" train.py --method %%M --seed 42 --headless
    if errorlevel 1 exit /b 1
)

endlocal
