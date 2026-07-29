$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".matplotlib-cache"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Create .venv and install requirements first."
}

& $Python (Join-Path $ProjectRoot "src\generate_data.py")
& $Python (Join-Path $ProjectRoot "src\explore_data.py")
& $Python (Join-Path $ProjectRoot "src\train_baseline.py")
& $Python (Join-Path $ProjectRoot "src\train_random_forest.py")
& $Python (Join-Path $ProjectRoot "src\analyze_models.py")
& $Python -m unittest discover (Join-Path $ProjectRoot "tests")

Write-Output "Pipeline completed successfully."

