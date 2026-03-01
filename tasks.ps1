param(
  [Parameter(Position=0)]
  [ValidateSet("help","venv","install","precommit","fmt","lint","type","test","test-fast","test-slow","check","aggregate-dry-run","clean")]
  [string]$Task = "help"
  ,
  [string]$AggInput = "data/scrape/scrape_TOOL_A.csv"
  ,
  [string]$AggConfig = "src/portfolio_fdc/configs/aggregate_tools.yaml"
  ,
  [string]$AggDetailOut = "data/detail"
)

$VenvDir = ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$PreCommit = Join-Path $VenvDir "Scripts\pre-commit.exe"

function Ensure-Venv {
  if (-not (Test-Path $Py)) {
    Write-Host "Creating venv in $VenvDir ..."
    python -m venv $VenvDir
  }
  & $Pip install -U pip | Out-Host
}

function Ensure-DevInstall {
  Ensure-Venv
  Write-Host "Installing dev dependencies ..."
  & $Pip install -e ".[dev]" | Out-Host
}

switch ($Task) {
  "help" {
    Write-Host "Tasks:"
    Write-Host "  .\tasks.ps1 venv       - create venv"
    Write-Host "  .\tasks.ps1 install    - install editable + dev deps"
    Write-Host "  .\tasks.ps1 precommit  - install pre-commit hooks"
    Write-Host "  .\tasks.ps1 fmt        - ruff format (apply)"
    Write-Host "  .\tasks.ps1 lint       - ruff check"
    Write-Host "  .\tasks.ps1 type       - mypy src"
    Write-Host "  .\tasks.ps1 test       - pytest"
    Write-Host "  .\tasks.ps1 test-fast  - pytest excluding slow tests"
    Write-Host "  .\tasks.ps1 test-slow  - pytest only slow tests"
    Write-Host "  .\tasks.ps1 check      - lint + type + test"
    Write-Host "  .\tasks.ps1 aggregate-dry-run - run aggregate without DB POST"
    Write-Host "  .\tasks.ps1 clean      - remove caches/build artifacts"
  }

  "venv" {
    Ensure-Venv
  }

  "install" {
    Ensure-DevInstall
  }

  "precommit" {
    Ensure-DevInstall
    Write-Host "Installing pre-commit hooks ..."
    & $PreCommit install | Out-Host
  }

  "fmt" {
    Ensure-DevInstall
    & $Py -m ruff format . | Out-Host
  }

  "lint" {
    Ensure-DevInstall
    & $Py -m ruff check . | Out-Host
  }

  "type" {
    Ensure-DevInstall
    & $Py -m mypy src | Out-Host
  }

  "test" {
    Ensure-DevInstall
    & $Py -m pytest | Out-Host
  }

  "test-fast" {
    Ensure-DevInstall
    & $Py -m pytest -m "not slow" | Out-Host
  }

  "test-slow" {
    Ensure-DevInstall
    & $Py -m pytest -m slow | Out-Host
  }

  "check" {
    Ensure-DevInstall
    & $Py -m ruff check . | Out-Host
    & $Py -m mypy src | Out-Host
    & $Py -m pytest | Out-Host
  }

  "aggregate-dry-run" {
    Ensure-DevInstall
    & $Py -m portfolio_fdc.main.aggregate --input $AggInput --config $AggConfig --detail-out $AggDetailOut --dry-run | Out-Host
  }

  "clean" {
    Write-Host "Cleaning caches/build artifacts ..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".pytest_cache"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".mypy_cache"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".ruff_cache"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "dist"
    Get-ChildItem -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Force -ErrorAction SilentlyContinue ".coverage"
  }
}
