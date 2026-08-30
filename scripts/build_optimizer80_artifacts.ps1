[CmdletBinding()]
param(
    [string]$StateBenchRoot = "I:\AI科研\Agentic\STATE-Bench",
    [string]$EnvFile = "I:\AI科研\Agentic\STATE-Bench\.env",
    [string]$OutputDirectory = "artifacts/statebench_cross_domain_pwm/memory",
    [string]$WorkRoot = "outputs/optimizer80-build",
    [int]$Workers = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Environment file not found: $Path"
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Wait-Relay([int]$Port, [int]$Seconds = 20) {
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Relay did not start on 127.0.0.1:$Port"
}

if ($Workers -lt 1 -or $Workers -gt 2) {
    throw "The optimizer80 build is locked to one or two workers"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$stateBench = (Resolve-Path -LiteralPath $StateBenchRoot).Path
$python = Join-Path $stateBench ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "STATE-Bench Python is missing: $python"
}

$outputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
$repoPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $outputRoot.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must resolve inside the selective-PWM repository: $outputRoot"
}
$v1Output = Join-Path $outputRoot "process_workflows_optimizer80.json"
$v2Output = Join-Path $outputRoot "workflow_router_v2_optimizer80.json"
foreach ($path in ($v1Output, $v2Output)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite an optimizer artifact: $path"
    }
}

Import-DotEnv $EnvFile
if ($env:STATE_BENCH_AGENT_MODEL -cne "gpt-5.4") {
    throw "The optimizer80 builder requires STATE_BENCH_AGENT_MODEL=gpt-5.4"
}
if (-not $env:STATE_BENCH_AGENT_API_KEY -or -not $env:STATE_BENCH_AGENT_BASE_URL) {
    throw "The NovaCode agent API key and base URL must be configured"
}
if ($env:STATE_BENCH_AGENT_BASE_URL -match '^http://(?:127\.0\.0\.1|localhost):8765') {
    throw "The environment file must contain the real NovaCode upstream, not the local relay"
}

$port = 8765
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $port is already in use"
}

$workDirectory = [IO.Path]::GetFullPath((Join-Path $repoRoot $WorkRoot))
if (-not $workDirectory.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "WorkRoot must resolve inside the selective-PWM repository: $workDirectory"
}
$runDirectory = Join-Path $workDirectory ("run-" + [Guid]::NewGuid().ToString("N"))
$cacheDirectory = Join-Path $workDirectory "workflow_cache"
New-Item -ItemType Directory -Force -Path $runDirectory, $cacheDirectory, $outputRoot | Out-Null
$stagedV1 = Join-Path $runDirectory "process_workflows_optimizer80.json"
$stagedV2 = Join-Path $runDirectory "workflow_router_v2_optimizer80.json"

$env:SHIM_UPSTREAM = $env:STATE_BENCH_AGENT_BASE_URL.TrimEnd("/")
$env:SHIM_PORT = [string]$port
$env:SHIM_RPM = "45"
$env:SHIM_BURST = "5"
$env:SHIM_BURST_WINDOW = "1.0"
$env:SHIM_ATTEMPTS = "5"
$env:SHIM_TIMEOUT = "600"
$env:SHIM_VERBOSE = "1"
$env:SHIM_LEDGER_PATH = Join-Path $runDirectory "relay.jsonl"
$env:WORKFLOW_LLM_API_KEY = $env:STATE_BENCH_AGENT_API_KEY
$env:NO_PROXY = (($env:NO_PROXY, "127.0.0.1", "localhost") -join ",").Trim(",")
$env:no_proxy = $env:NO_PROXY
$env:PYTHONUTF8 = "1"

$relay = Start-Process -FilePath $python `
    -ArgumentList @((Join-Path $repoRoot "tools\eval_shim.py")) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput (Join-Path $runDirectory "relay.log") `
    -RedirectStandardError (Join-Path $runDirectory "relay.err.log") `
    -WindowStyle Hidden `
    -PassThru

try {
    Wait-Relay -Port $port
    & $python (Join-Path $repoRoot "scripts\build_process_workflows.py") `
        --data-root (Join-Path $stateBench "datasets\train_task_trajectories") `
        --output $stagedV1 `
        --task-split optimizer `
        --task-manifest (Join-Path $repoRoot "configs\workflow_router_dev_ids.json") `
        --llm-base-url "http://127.0.0.1:$port/v1" `
        --llm-provider-tag novacode `
        --llm-model gpt-5.4 `
        --llm-workers $Workers `
        --llm-timeout 600 `
        --llm-max-retries 0 `
        --cache-dir $cacheDirectory
    if ($LASTEXITCODE -ne 0) { throw "optimizer80 v1 build failed with code $LASTEXITCODE" }

    & $python (Join-Path $repoRoot "scripts\build_workflow_router_v2.py") `
        --state-bench-root $stateBench `
        --dev-manifest (Join-Path $repoRoot "configs\workflow_router_dev_ids.json") `
        --memory-training-split optimizer `
        --v1-artifact $stagedV1 `
        --output $stagedV2 `
        --promoted-domains shopping_assistant
    if ($LASTEXITCODE -ne 0) { throw "optimizer80 v2 build failed with code $LASTEXITCODE" }

    & $python -c "import hashlib,json,pathlib,sys; v1=pathlib.Path(sys.argv[1]); v2=pathlib.Path(sys.argv[2]); a=json.loads(v1.read_text()); b=json.loads(v2.read_text()); assert len(a['cards']) > 0; assert b['provenance']['memory_training_split']=='optimizer'; assert b['provenance']['lockbox_independent'] is True; assert b['source_memory_sha256']==hashlib.sha256(v1.read_bytes()).hexdigest(); assert all(not x for d in b['provenance']['source_task_overlap'].values() for x in d.values()); print(json.dumps({'v1_cards':len(a['cards']),'v2_cards':len(b['cards']),'lockbox_independent':True}))" $stagedV1 $stagedV2
    if ($LASTEXITCODE -ne 0) { throw "optimizer80 cross-artifact validation failed" }

    Move-Item -LiteralPath $stagedV1 -Destination $v1Output
    Move-Item -LiteralPath $stagedV2 -Destination $v2Output
    Write-Host "Optimizer80 artifacts installed under $outputRoot"
} finally {
    if ($null -ne $relay -and -not $relay.HasExited) {
        Stop-Process -Id $relay.Id
    }
    Remove-Item Env:\WORKFLOW_LLM_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\SHIM_LEDGER_PATH -ErrorAction SilentlyContinue
}
