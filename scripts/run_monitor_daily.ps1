$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $rootDir ".venv\Scripts\python.exe"
$target = Join-Path $rootDir "main.py"

& $python $target
exit $LASTEXITCODE
