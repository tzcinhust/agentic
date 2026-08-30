[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LockboxBaseline,
    [Parameter(Mandatory = $true)]
    [string]$LockboxCandidate,
    [string]$StateBenchRoot = "I:\AI科研\Agentic\STATE-Bench",
    [string]$EnvFile = "I:\AI科研\Agentic\STATE-Bench\.env",
    [string]$OutputDirectory = "artifacts/statebench_cross_domain_pwm/memory",
    [string]$WorkRoot = "outputs/full100-build",
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
    throw "The full100 build is locked to one or two workers"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$stateBench = (Resolve-Path -LiteralPath $StateBenchRoot).Path
$python = Join-Path $stateBench ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "STATE-Bench Python is missing: $python"
}

$baselineRoot = (Resolve-Path -LiteralPath $LockboxBaseline).Path
$candidateRoot = (Resolve-Path -LiteralPath $LockboxCandidate).Path
$workDirectory = [IO.Path]::GetFullPath((Join-Path $repoRoot $WorkRoot))
$repoPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $workDirectory.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "WorkRoot must resolve inside the selective-PWM repository: $workDirectory"
}
$runDirectory = Join-Path $workDirectory ("run-" + [Guid]::NewGuid().ToString("N"))
$cacheDirectory = Join-Path $workDirectory "workflow_cache"
New-Item -ItemType Directory -Force -Path $runDirectory, $cacheDirectory | Out-Null

# Re-evaluate the immutable paired lockbox evidence under the current HEAD.
# A full-100 rebuild is forbidden unless the final cumulative C arm passes.
$gateReport = Join-Path $runDirectory "lockbox-gate.json"
& $python (Join-Path $repoRoot "scripts\evaluate_gate.py") `
    --gate lockbox `
    --baseline $baselineRoot `
    --candidate $candidateRoot `
    --state-bench-root $stateBench `
    --output $gateReport
if ($LASTEXITCODE -ne 0) {
    throw "Lockbox gate did not pass; refusing to read all 100 training trajectories"
}
$gate = Get-Content -LiteralPath $gateReport -Raw | ConvertFrom-Json
if ($gate.passed -ne $true -or $gate.gate -cne "lockbox" -or
    $gate.candidate_router_stage -cne "C") {
    throw "Full100 rebuild requires a passing cumulative C-stage lockbox report"
}

$outputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
if (-not $outputRoot.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDirectory must resolve inside the selective-PWM repository: $outputRoot"
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$v1Output = Join-Path $outputRoot "process_workflows.json"
$v2Output = Join-Path $outputRoot "workflow_router_v2.json"
$stagedV1 = Join-Path $runDirectory "process_workflows.json"
$stagedV2 = Join-Path $runDirectory "workflow_router_v2.json"

Import-DotEnv $EnvFile
if ($env:STATE_BENCH_AGENT_MODEL -cne "gpt-5.4") {
    throw "The full100 builder requires STATE_BENCH_AGENT_MODEL=gpt-5.4"
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
        --task-split all `
        --llm-base-url "http://127.0.0.1:$port/v1" `
        --llm-provider-tag novacode `
        --llm-model gpt-5.4 `
        --llm-workers $Workers `
        --llm-timeout 600 `
        --llm-max-retries 0 `
        --cache-dir $cacheDirectory
    if ($LASTEXITCODE -ne 0) { throw "full100 v1 build failed with code $LASTEXITCODE" }

    & $python (Join-Path $repoRoot "scripts\build_workflow_router_v2.py") `
        --state-bench-root $stateBench `
        --dev-manifest (Join-Path $repoRoot "configs\workflow_router_dev_ids.json") `
        --memory-training-split all `
        --v1-artifact $stagedV1 `
        --output $stagedV2 `
        --promoted-domains shopping_assistant
    if ($LASTEXITCODE -ne 0) { throw "full100 v2 build failed with code $LASTEXITCODE" }

    & $python (Join-Path $repoRoot "scripts\preflight_training_artifacts.py") `
        --kind full100 `
        --memory $stagedV1 `
        --router $stagedV2 `
        --state-bench-root $stateBench `
        --repository-root $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "full100 provenance preflight failed" }

    foreach ($path in ($v1Output, $v2Output)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Copy-Item -LiteralPath $path -Destination (Join-Path $runDirectory ((Split-Path $path -Leaf) + ".previous"))
        }
    }
    Move-Item -LiteralPath $stagedV1 -Destination $v1Output -Force
    Move-Item -LiteralPath $stagedV2 -Destination $v2Output -Force
    Write-Host "Full100 artifacts installed under $outputRoot; prior files were backed up in $runDirectory"
} finally {
    if ($null -ne $relay -and -not $relay.HasExited) {
        Stop-Process -Id $relay.Id
    }
    Remove-Item Env:\WORKFLOW_LLM_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\SHIM_LEDGER_PATH -ErrorAction SilentlyContinue
}
