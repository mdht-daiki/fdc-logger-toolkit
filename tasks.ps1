param(
  [Parameter(Position=0)]
  [ValidateSet("help","venv","install","precommit","fmt","lint","type","test","check","clean")]
  [string]$Task = "help"
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
    Write-Host "  .\tasks.ps1 check      - lint + type + test"
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

  "check" {
    Ensure-DevInstall
    & $Py -m ruff check . | Out-Host
    & $Py -m mypy src | Out-Host
    & $Py -m pytest | Out-Host
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
