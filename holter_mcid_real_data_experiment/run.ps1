param(
    [string]$Holter = "",
    [string]$Clinical = "",
    [string]$Cohort = "",
    [string]$OutputDir = ""
)

$experimentDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = Split-Path -Parent $experimentDir
$python = Join-Path $workspaceDir "experiment_1_information_source_ablation\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found at $python"
}

if ([string]::IsNullOrWhiteSpace($Holter)) {
    $Holter = Join-Path $workspaceDir "data\paired_holter.csv"
}
if ([string]::IsNullOrWhiteSpace($Clinical)) {
    $Clinical = Join-Path $workspaceDir "data\clinical_anchor.csv"
}
if ([string]::IsNullOrWhiteSpace($Cohort)) {
    $Cohort = Join-Path $workspaceDir "data\cohort.csv"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $workspaceDir "results\holter_mcid_real"
}

if (-not (Test-Path -LiteralPath $Holter)) {
    throw "Holter input not found at $Holter"
}

$arguments = @(
    (Join-Path $experimentDir "holter_mcid_real_data_analysis.py"),
    "--holter", $Holter,
    "--output-dir", $OutputDir,
    "--bootstrap", "2000",
    "--seed", "20260818"
)

if (Test-Path -LiteralPath $Clinical) {
    $arguments += @("--clinical", $Clinical)
}
if (Test-Path -LiteralPath $Cohort) {
    $arguments += @("--cohort", $Cohort)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Holter MCID analysis failed with exit code $LASTEXITCODE"
}
