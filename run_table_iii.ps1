param(
    [int]$Episodes = 1000,
    [int]$Seed = 42,
    [string]$OutputDir = "results/table_iii"
)

# Later stage: evaluate all 12 deployment configurations from Table III.
$configs = @(
    "nominal",
    "dual_runtime_filter",
    "dual_no_runtime_filter",
    "reward_only",
    "filter_only_runtime_filter",
    "filter_only_no_runtime_filter",
    "nominal_dr",
    "dual_dr_runtime_filter",
    "dual_dr_no_runtime_filter",
    "reward_only_dr",
    "filter_only_dr_runtime_filter",
    "filter_only_dr_no_runtime_filter"
)

foreach ($configName in $configs) {
    python test.py --config $configName --episodes $Episodes --seed $Seed --output-dir $OutputDir --headless
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $configName"
    }
}
