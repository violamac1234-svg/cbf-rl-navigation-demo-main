@echo off
setlocal
set "PYTHON_EXE=D:\anaconda3\envs\cbf_learning\python.exe"

rem Stage 2: train all four 20-percent dynamics-randomization models.
for %%M in (nominal reward_only filter_only dual) do (
    "%PYTHON_EXE%" train.py --method %%M --seed 42 --domain-randomization --headless
    if errorlevel 1 exit /b 1
)

endlocal
