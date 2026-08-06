param(
    [int]$Seed = 42
)

# Stage 2: train the separate 20-percent dynamics-randomization variants.
$methods = @("nominal", "reward_only", "filter_only", "dual")

foreach ($methodName in $methods) {
    python train.py --method $methodName --seed $Seed --domain-randomization --headless
    if ($LASTEXITCODE -ne 0) {
        throw "DR training failed for $methodName"
    }
}
