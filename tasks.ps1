param(
  [Parameter(Position=0)]
  [ValidateSet("help","venv","install","precommit","fmt","lint","type","test","test-fast","test-slow","test-db-api-aggregate","check","aggregate-dry-run","clean","cleanup-retention","vacuum-db")]
  [string]$Task = "help"
  ,
  [string]$AggInput = "data/scrape/scrape_TOOL_A.csv"
  ,
  [string]$AggConfig = "src/portfolio_fdc/configs/aggregate_tools.yaml"
  ,
  [string]$AggDetailOut = "data/detail"
)

$RepoRoot = $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
$PreCommit = Join-Path $VenvDir "Scripts\pre-commit.exe"
# Mirror db.py: use PORTFOLIO_DB_DIR env var when set, otherwise fall back to <repo>/data/db
$DbDir = if ($env:PORTFOLIO_DB_DIR) { $env:PORTFOLIO_DB_DIR } else { Join-Path $RepoRoot "data\db" }
$DbPath = Join-Path $DbDir "main.db"

function Ensure-Venv {
  param(
    [switch]$SkipPipUpgrade
  )

  if (-not (Test-Path $Py)) {
    if ($SkipPipUpgrade) {
      Write-Error "Venv not found at '$Py' (VenvDir: '$VenvDir'). Run '.\tasks.ps1 install' first before executing scheduled tasks."
      exit 1
    }
    Write-Host "Creating venv in $VenvDir ..."
    python -m venv $VenvDir
  }

  if (-not $SkipPipUpgrade) {
    & $Pip install -U pip | Out-Host
  }
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
    Write-Host "  .\tasks.ps1 test-db-api-aggregate - pytest for db_api + aggregate connection"
    Write-Host "  .\tasks.ps1 check      - lint + type + test"
    Write-Host "  .\tasks.ps1 aggregate-dry-run - run aggregate without DB POST"
    Write-Host "  .\tasks.ps1 clean      - remove caches/build artifacts"
    Write-Host "  .\tasks.ps1 cleanup-retention - delete retention-expired data (daily task)"
    Write-Host "  .\tasks.ps1 vacuum-db - VACUUM SQLite database (weekly task)"
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

  "test-db-api-aggregate" {
    Ensure-DevInstall
    & $Py -m pytest tests/test_db_api_integration.py tests/test_aggregate_db_api_integration.py | Out-Host
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

  "cleanup-retention" {
    # Daily retention cleanup task for Windows Task Scheduler
    # Deletes records older than retention period (see docs/decision-log.md論点9)
    # Execution order: child -> parent to maintain FK integrity
    Write-Host "Starting retention cleanup ..."
    $StartTime = Get-Date
    $LogPath = Join-Path $RepoRoot ("data\logs\cleanup_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

    # Create log directory if not exists
    $LogDir = Split-Path $LogPath
    if (-not (Test-Path $LogDir)) {
      New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }

    try {
      # Scheduled jobs should rely on a pre-provisioned venv; dependency updates stay manual via the install task.
      Ensure-Venv -SkipPipUpgrade
      # Guard: fail early if the module is not yet implemented
      & $Py -c "import portfolio_fdc.tools.retention_cleanup" 2>$null
      if ($LASTEXITCODE -ne 0) {
        Write-Host "NOT IMPLEMENTED: portfolio_fdc.tools.retention_cleanup is not available. Deploy the module before enabling this task." -ForegroundColor Yellow
        exit 1
      }
      # TODO: Implement cleanup SQL via db_api or direct SQLite query
      # Expected: Delete rows from child tables first (StepWindows, Parameters, ChartsHistory,
      # governance tables) and then parent ProcessInfo based on retention policy (実データ 1年, 監査系 3年)
      & $Py -m portfolio_fdc.tools.retention_cleanup --db-path $DbPath --log-file $LogPath | Out-Host
      if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: retention_cleanup exited with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
      }

      $EndTime = Get-Date
      $Duration = ($EndTime - $StartTime).TotalSeconds
      Write-Host "Retention cleanup completed in $Duration seconds"
      exit 0
    }
    catch {
      Write-Host "ERROR: Retention cleanup failed: $_" -ForegroundColor Red
      exit 1
    }
  }

  "vacuum-db" {
    # Weekly VACUUM task for Windows Task Scheduler
    # Reclaims physical disk space after retention cleanup
    # MUST run AFTER cleanup task to avoid lock contention
    Write-Host "Starting database VACUUM ..."
    $StartTime = Get-Date
    $LogPath = Join-Path $RepoRoot ("data\logs\vacuum_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

    # Create log directory if not exists
    $LogDir = Split-Path $LogPath
    if (-not (Test-Path $LogDir)) {
      New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }

    try {
      # Scheduled jobs should rely on a pre-provisioned venv; dependency updates stay manual via the install task.
      Ensure-Venv -SkipPipUpgrade
      # Guard: fail early if the module is not yet implemented
      & $Py -c "import portfolio_fdc.tools.vacuum_database" 2>$null
      if ($LASTEXITCODE -ne 0) {
        Write-Host "NOT IMPLEMENTED: portfolio_fdc.tools.vacuum_database is not available. Deploy the module before enabling this task." -ForegroundColor Yellow
        exit 1
      }
      # TODO: Implement VACUUM via sqlite3 CLI or direct API call
      # Expected: Execute PRAGMA optimize; followed by VACUUM to reclaim disk space
      & $Py -m portfolio_fdc.tools.vacuum_database --db-path $DbPath --log-file $LogPath | Out-Host
      if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: vacuum_database exited with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
      }

      $EndTime = Get-Date
      $Duration = ($EndTime - $StartTime).TotalSeconds
      Write-Host "Database VACUUM completed in $Duration seconds"
      exit 0
    }
    catch {
      Write-Host "ERROR: Database VACUUM failed: $_" -ForegroundColor Red
      exit 1
    }
  }
}
