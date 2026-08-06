param(
    [int]$Seed = 42,
    [int]$NumEnvs = 1024,
    [int]$Iterations = 1000,
    [string]$PythonExecutable = "D:\anaconda3\envs\cbf_learning\python.exe"
)

$methods = @("nominal", "reward_only", "filter_only", "dual")
foreach ($methodName in $methods) {
    & $PythonExecutable train_arm2d.py --method $methodName --seed $Seed --num-envs $NumEnvs --iterations $Iterations
    if ($LASTEXITCODE -ne 0) {
        throw "Arm2D training failed for $methodName"
    }
}
