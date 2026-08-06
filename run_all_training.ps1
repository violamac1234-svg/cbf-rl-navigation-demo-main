param(
    [int]$Seed = 42,
    [string]$PythonExecutable = "python"
)

# Stage 1: train only the four base ablation methods, without DR.
$methods = @("nominal", "reward_only", "filter_only", "dual")

foreach ($methodName in $methods) {
    & $PythonExecutable train.py --method $methodName --seed $Seed --headless
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $methodName"
    }
}
