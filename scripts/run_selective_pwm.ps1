[CmdletBinding()]
param(
    [ValidateSet("dev", "lockbox", "paired150", "official750")]
    [string]$Stage = "dev",

    [ValidateSet("baseline", "candidate")]
    [string]$Arm = "candidate",

    [ValidateSet("A", "B", "C")]
    [string]$RouterStage = "C",

    [string[]]$Domains = @("shopping_assistant", "travel", "customer_support"),
    [string]$StateBenchRoot = "I:\AI科研\Agentic\STATE-Bench",
    [string]$EnvFile = "I:\AI科研\Agentic\STATE-Bench\.env",
    [string]$OutputRoot = "outputs/selective_pwm",
    [int]$RunStart = 1,
    [int]$Workers = 2,
    [switch]$StartRelay,
    [switch]$Resume
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
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($listener) { return }
        Start-Sleep -Milliseconds 250
    }
    throw "Relay did not start on 127.0.0.1:$Port"
}

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot hash missing file: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256([string]$Text) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([Convert]::ToHexString($sha.ComputeHash($bytes))).ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-CanonicalJsonHash($Value) {
    # Python's sorted compact JSON is also used by the validators.  The payload
    # intentionally contains no timestamp, so an identical invocation hashes
    # identically and can be compared safely on -Resume.
    $json = $Value | ConvertTo-Json -Depth 40 -Compress
    $hash = $json | & $script:stateBenchPython -c "import hashlib,json,sys; value=json.load(sys.stdin); raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'); print(hashlib.sha256(raw).hexdigest())"
    if ($LASTEXITCODE -ne 0 -or -not $hash) { throw "Failed to compute canonical manifest hash" }
    return ([string]$hash).Trim().ToLowerInvariant()
}

function Write-JsonExclusive([string]$Path, $Value) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite JSON input: $Path" }
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $json = (ConvertTo-Json -InputObject $Value -Depth 40) + "`n"
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally {
        $stream.Dispose()
    }
}

function Remove-TemporaryFile([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        (Get-Item -LiteralPath $Path).IsReadOnly = $false
        Remove-Item -LiteralPath $Path -Force
    }
}

function Get-FileLengthOrZero([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return 0L }
    return [long](Get-Item -LiteralPath $Path).Length
}

function Invoke-CapturedNative(
    [string]$Executable,
    [object[]]$Arguments,
    [string]$LogPath
) {
    if (Test-Path -LiteralPath $LogPath) { throw "Refusing to overwrite immutable batch log: $LogPath" }
    $parent = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    & $Executable @Arguments 2>&1 |
        Tee-Object -FilePath $LogPath |
        ForEach-Object { Write-Host $_ }
    $exitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        [IO.File]::WriteAllText($LogPath, "", [Text.UTF8Encoding]::new($false))
    }
    return [int]$exitCode
}

function New-AuditLogPath(
    [string]$ArmRoot,
    [string]$Domain,
    [int]$RunIndex,
    [string]$Mode
) {
    $id = [Guid]::NewGuid().ToString("N")
    return Join-Path $ArmRoot "_batch_logs\$Domain\run$RunIndex\$Mode-$id.log"
}

function Write-AuditNoteExclusive([string]$Path, [string]$Message) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite immutable audit log: $Path" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Message + "`n")
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally {
        $stream.Dispose()
    }
}

function New-TaskIdFile([string[]]$TaskIds) {
    $path = Join-Path ([IO.Path]::GetTempPath()) "selective-pwm-task-ids-$([Guid]::NewGuid().ToString('N')).json"
    Write-JsonExclusive -Path $path -Value @($TaskIds)
    return $path
}

function New-RunSnapshot(
    [string]$ResumeHelper,
    [string]$RunDirectory,
    [string]$TaskIdsPath
) {
    $path = Join-Path ([IO.Path]::GetTempPath()) "selective-pwm-snapshot-$([Guid]::NewGuid().ToString('N')).json"
    & $script:stateBenchPython $ResumeHelper snapshot `
        --run-dir $RunDirectory `
        --task-ids-json $TaskIdsPath `
        --output $path
    if ($LASTEXITCODE -ne 0) { throw "Failed to capture immutable pre-session state" }
    return $path
}

function Write-AuditSession(
    [string]$ResumeHelper,
    [string]$ArmRoot,
    [string]$Domain,
    [int]$RunIndex,
    [string]$RunManifest,
    [string]$Mode,
    [string]$AllTaskIdsPath,
    [string]$TargetTaskIdsPath,
    [string]$PreSnapshot,
    [string]$LogPath,
    [string]$RelayLedger,
    [long]$RelayStartOffset,
    [int]$ProcessExitCode
) {
    $args = @(
        $ResumeHelper, "record",
        "--arm-root", $ArmRoot,
        "--domain", $Domain,
        "--run-index", [string]$RunIndex,
        "--run-manifest", $RunManifest,
        "--mode", $Mode,
        "--task-ids-json", $AllTaskIdsPath,
        "--pre-snapshot", $PreSnapshot,
        "--log", $LogPath,
        "--relay-ledger", $RelayLedger,
        "--relay-start-offset", [string]$RelayStartOffset,
        "--process-exit-code", [string]$ProcessExitCode
    )
    if ($TargetTaskIdsPath) { $args += @("--target-task-ids-json", $TargetTaskIdsPath) }
    & $script:stateBenchPython @args
    if ($LASTEXITCODE -ne 0) { throw "Failed to append the auditable session record" }
}

function Write-AuditOfficialBatchSession(
    [string]$ResumeHelper,
    [string]$ArmRoot,
    [string]$Domain,
    [string]$RunManifest,
    [string]$AllTaskIdsPath,
    [string]$PreSnapshotsJson,
    [string]$LogPath,
    [string]$RelayLedger,
    [long]$RelayStartOffset,
    [int]$ProcessExitCode
) {
    & $script:stateBenchPython $ResumeHelper record-official-batch `
        --arm-root $ArmRoot `
        --domain $Domain `
        --run-manifest $RunManifest `
        --task-ids-json $AllTaskIdsPath `
        --pre-snapshots-json $PreSnapshotsJson `
        --log $LogPath `
        --relay-ledger $RelayLedger `
        --relay-start-offset $RelayStartOffset `
        --process-exit-code $ProcessExitCode `
        --split test `
        --num-runs 5 `
        --num-runs-idx-start 1
    if ($LASTEXITCODE -ne 0) { throw "Failed to append official fresh-batch audit records" }
}

function Get-ResumePlan(
    [string]$ResumeHelper,
    [string]$ArmRoot,
    [string]$Domain,
    [int]$RunIndex,
    [string]$RunManifest,
    [string]$TaskIdsPath
) {
    $raw = & $script:stateBenchPython $ResumeHelper plan `
        --arm-root $ArmRoot `
        --domain $Domain `
        --run-index $RunIndex `
        --run-manifest $RunManifest `
        --task-ids-json $TaskIdsPath
    if ($LASTEXITCODE -ne 0) { throw "Resume evidence validation failed closed for $Domain run$RunIndex" }
    return ($raw | ConvertFrom-Json)
}

function Get-EvaluationDeploymentMap {
    $all = [Environment]::GetEnvironmentVariables("Process")
    $names = @(
        $all.Keys |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -match '^STATE_BENCH_EVAL_DEPLOYMENTS(?:_\d+)?$' } |
            Sort-Object
    )
    $numbered = @($names | Where-Object { $_ -match '_\d+$' })
    if ($numbered.Count -gt 0) {
        throw (
            "Numbered STATE_BENCH_EVAL_DEPLOYMENTS/ENDPOINT pools are forbidden because they can bypass " +
            "the single attributable relay: $($numbered -join ', ')"
        )
    }
    $numberedEndpoints = @(
        $all.Keys |
            ForEach-Object { [string]$_ } |
            Where-Object { $_ -match '^STATE_BENCH_EVAL_ENDPOINT_\d+$' } |
            Sort-Object
    )
    if ($numberedEndpoints.Count -gt 0) {
        throw "Numbered STATE_BENCH_EVAL_ENDPOINT pools are forbidden: $($numberedEndpoints -join ', ')"
    }
    $result = [ordered]@{}
    foreach ($name in $names) {
        $values = @(
            ([string]$all[$name]).Split(',') |
                ForEach-Object { $_.Trim() } |
                Where-Object { $_ }
        )
        if ($values.Count -gt 0) { $result[$name] = $values }
    }
    $deployments = @($result.Values | ForEach-Object { @($_) })
    if ($deployments.Count -eq 0) {
        throw "Set at least one STATE_BENCH_EVAL_DEPLOYMENTS variable to gpt-5.4"
    }
    $wrong = @($deployments | Where-Object { $_ -cne 'gpt-5.4' } | Sort-Object -Unique)
    if ($wrong.Count -gt 0) {
        throw "Every primary/numbered evaluation deployment must be exactly gpt-5.4; found: $($wrong -join ', ')"
    }
    return $result
}

function Get-StateBenchVersion([string]$Root) {
    $pyprojectPath = Join-Path $Root "pyproject.toml"
    $content = Get-Content -LiteralPath $pyprojectPath -Raw
    $match = [regex]::Match($content, '(?m)^\s*version\s*=\s*"([^"]+)"')
    if (-not $match.Success) { throw "Cannot read STATE-Bench version from $pyprojectPath" }
    return $match.Groups[1].Value
}

function Assert-NoRootPythonShadows(
    [string]$Root,
    [string[]]$AllowedPackageDirectories
) {
    # Python imports root-level modules before site-packages.  Inspect the two
    # explicit runtime roots without invoking Python, so even sitecustomize.py
    # cannot execute before it is rejected.
    $shadowFiles = @(
        "sitecustomize.py", "usercustomize.py", "openai.py", "httpx.py",
        "state_bench.py", "agents.py", "clients.py", "tools.py",
        "site.py", "os.py", "sys.py", "json.py", "hashlib.py",
        "importlib.py", "pathlib.py", "typing.py", "uuid.py"
    )
    foreach ($name in $shadowFiles) {
        $candidate = Join-Path $Root $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            throw "Root-level Python shadow file is forbidden: $candidate"
        }
    }

    $shadowDirectories = @(
        "sitecustomize", "usercustomize", "openai", "httpx",
        "state_bench", "agents", "clients", "tools"
    )
    foreach ($name in $shadowDirectories) {
        if ($name -in $AllowedPackageDirectories) { continue }
        $candidate = Join-Path $Root $name
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            throw "Unexpected root-level Python shadow package is forbidden: $candidate"
        }
    }
}

function Assert-PythonModuleOrigins([string]$RepositoryRoot, [string]$BenchmarkRoot) {
    # The PowerShell-only root scan runs before this interpreter starts.  This
    # verifier deliberately uses normal site-package semantics so its import
    # resolution matches the actual STATE-Bench and relay processes.
    $checker = @'
import importlib.util
import json
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
benchmark = Path(sys.argv[2]).resolve()
expected_pythonpath = os.pathsep.join((str(repository), str(benchmark)))
errors = []
if os.environ.get("PYTHONPATH") != expected_pythonpath:
    errors.append("PYTHONPATH does not equal the two frozen runtime roots")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    errors.append("PYTHONNOUSERSITE is not enabled")
if os.environ.get("PYTHONSAFEPATH") != "1":
    errors.append("PYTHONSAFEPATH is not enabled")

targets = {
    "state_bench": benchmark / "state_bench",
    "agents": repository / "agents",
    "clients": repository / "clients",
    "tools.eval_shim": repository / "tools",
}


def is_within(path: Path, root: Path) -> bool:
    path_text = os.path.normcase(str(path.resolve()))
    root_text = os.path.normcase(str(root.resolve()))
    try:
        return os.path.commonpath((path_text, root_text)) == root_text
    except ValueError:
        return False


report = {}
for module_name, expected_root in targets.items():
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        spec = None
        errors.append(f"{module_name}: resolution raised {type(exc).__name__}")
    resolved = []
    if spec is not None:
        if spec.origin not in (None, "built-in", "frozen"):
            resolved.append(Path(spec.origin).resolve())
        if spec.submodule_search_locations:
            resolved.extend(Path(value).resolve() for value in spec.submodule_search_locations)
    if not resolved:
        errors.append(f"{module_name}: no filesystem-backed import location")
    elif not all(is_within(value, expected_root) for value in resolved):
        errors.append(f"{module_name}: resolved outside {expected_root}")
    report[module_name] = [str(value) for value in resolved]

eval_shim_origins = report.get("tools.eval_shim", [])
expected_eval_shim = (repository / "tools" / "eval_shim.py").resolve()
if len(eval_shim_origins) != 1 or Path(eval_shim_origins[0]) != expected_eval_shim:
    errors.append("tools.eval_shim did not resolve to the frozen relay implementation")

if errors:
    print(json.dumps({"errors": errors, "resolved": report}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)
print(json.dumps(report, ensure_ascii=False, sort_keys=True))
'@
    $result = @(& $script:stateBenchPython -c $checker $RepositoryRoot $BenchmarkRoot 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Python module provenance preflight failed: $($result -join ' ')"
    }
    Write-Host "Python module provenance preflight passed: $($result -join ' ')"
}

function New-RunManifest(
    [string]$Domain,
    [string[]]$TaskIds,
    [string]$TaskSource,
    [string]$TaskMode,
    [string]$AgentClass,
    [int]$Runs,
    [string]$MemoryPath,
    [string]$RouterPath,
    [string]$ArtifactKind,
    [string]$RunnerPath,
    [string]$RepositoryCommit,
    [bool]$RepositoryTrackedTreeClean,
    $ImplementationHashes,
    $TransportEvidence,
    [string]$StateBenchCommit,
    [string]$StateBenchVersion,
    [string]$ProtocolPath,
    $Protocol,
    [string]$SplitManifestPath,
    $EvaluationDeployments,
    [bool]$StateBenchTrackedTreeClean
) {
    $taskText = ((@($TaskIds) -join "`n") + "`n")
    $core = [ordered]@{
        schema_version = "1.0.0"
        created_by = "scripts/run_selective_pwm.ps1"
        stage = $Stage
        arm = $Arm
        router_stage = if ($Arm -eq "candidate") { $RouterStage } else { $null }
        domain = $Domain
        protocol = [ordered]@{
            benchmark_version = "0.8.1"
            evaluation_protocol_id = "state_bench_v0.8.1_gpt54"
            split_version = [string]$Protocol.split_version
            official_split = [string]$Protocol.split
            official_num_runs = [int]$Protocol.num_runs
            agent_model = "gpt-5.4"
            simulator_model = [string]$Protocol.simulator.model
            judge_model = [string]$Protocol.judge.model
            judge_reasoning_effort = [string]$Protocol.judge.reasoning_effort
            protocol_config_sha256 = Get-FileSha256 $ProtocolPath
            prompt_hashes = [ordered]@{
                simulator = $Protocol.simulator.prompt_hashes
                judge = $Protocol.judge.prompt_hashes
            }
            evaluation_deployments = $EvaluationDeployments
        }
        run = [ordered]@{
            num_runs = $Runs
            run_start = $RunStart
            workers = $Workers
            retry_attempts = 1
            retrieve_learnings_top_k = 3
            ignore_missing_runs = $false
            agent_class = $AgentClass
            agent_client_class = "OpenCodeLLMClient"
            memory_mode = "hybrid"
            agent_client_contract = [ordered]@{
                max_tokens = 4096
                timeout_seconds = 120
                max_retries = 0
                temperature = 0
            }
            official_evaluation_client_contract = [ordered]@{
                openai_sdk_version = "2.16.0"
                openai_sdk_default_max_retries = 2
                benchmark_tenacity_max_attempts = 5
                configuration_source = "pinned_state_bench_v0.8.1"
                all_requests_via_attributable_relay = $true
                scored_trajectory_resampling = $false
            }
            task_selection = [ordered]@{
                mode = $TaskMode
                source = $TaskSource
                split = if ($TaskMode -eq "split") { "test" } else { $null }
                task_ids = @($TaskIds)
                task_ids_sha256 = Get-TextSha256 $taskText
            }
        }
        artifacts = [ordered]@{
            artifact_kind = $ArtifactKind
            memory_sha256 = Get-FileSha256 $MemoryPath
            router_sha256 = Get-FileSha256 $RouterPath
            runner_sha256 = Get-FileSha256 $RunnerPath
            repository_commit = $RepositoryCommit
            repository_tracked_tree_clean = $RepositoryTrackedTreeClean
            implementation_sha256 = $ImplementationHashes
            state_bench_commit = $StateBenchCommit
            state_bench_version = $StateBenchVersion
            state_bench_tracked_tree_clean = $StateBenchTrackedTreeClean
            state_bench_protocol_sha256 = Get-FileSha256 $ProtocolPath
            state_bench_split_manifest_sha256 = Get-FileSha256 $SplitManifestPath
        }
        transport = $TransportEvidence
    }
    $manifest = [ordered]@{}
    foreach ($key in $core.Keys) { $manifest[$key] = $core[$key] }
    $manifest["manifest_sha256"] = Get-CanonicalJsonHash $core
    return $manifest
}

function Write-RunManifestExclusive([string]$Path, $Expected, [switch]$IsResume) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        if (-not $IsResume) { throw "Run manifest already exists: $Path" }
        $stored = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $storedCore = [ordered]@{}
        foreach ($property in $stored.PSObject.Properties) {
            if ($property.Name -ne "manifest_sha256") { $storedCore[$property.Name] = $property.Value }
        }
        $storedSelfHash = Get-CanonicalJsonHash $storedCore
        if ($stored.manifest_sha256 -ne $storedSelfHash) {
            throw "Existing run manifest has an invalid self-hash: $Path"
        }
        if ($stored.manifest_sha256 -ne $Expected.manifest_sha256) {
            throw "-Resume invocation does not match the immutable run manifest: $Path"
        }
        return
    }
    if ($IsResume) { throw "-Resume requires an existing immutable run manifest: $Path" }
    $temporary = "$Path.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
    try {
        $json = ($Expected | ConvertTo-Json -Depth 40) + "`n"
        [IO.File]::WriteAllText($temporary, $json, [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temporary, $Path)
    } finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$stateBench = (Resolve-Path -LiteralPath $StateBenchRoot).Path
$script:stateBenchPython = Join-Path $stateBench ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $script:stateBenchPython -PathType Leaf)) {
    throw "STATE-Bench Python is missing: $script:stateBenchPython"
}
Import-DotEnv $EnvFile
$env:PYTHONPATH = (($repoRoot, $stateBench) -join [IO.Path]::PathSeparator)
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONSAFEPATH = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONHOME = $null
$env:PYTHONSTARTUP = $null
$env:PYTHONINSPECT = $null
Assert-NoRootPythonShadows `
    -Root $repoRoot `
    -AllowedPackageDirectories @("agents", "clients", "tools")
Assert-NoRootPythonShadows `
    -Root $stateBench `
    -AllowedPackageDirectories @("state_bench", "agents", "clients")
Assert-PythonModuleOrigins -RepositoryRoot $repoRoot -BenchmarkRoot $stateBench
$evaluationRetryProbe = @'
import json, pathlib, sys
import openai, yaml
from openai._constants import DEFAULT_MAX_RETRIES
config = yaml.safe_load((pathlib.Path(sys.argv[1]) / "state_bench" / "configs" / "llm.yaml").read_text(encoding="utf-8"))
print(json.dumps({"openai_sdk_version": openai.__version__, "openai_sdk_default_max_retries": DEFAULT_MAX_RETRIES, "benchmark_tenacity_max_attempts": config["retry"]["max_attempts"]}, separators=(",", ":")))
'@
$evaluationRetryRuntime = & $script:stateBenchPython -c $evaluationRetryProbe $stateBench
if ($LASTEXITCODE -ne 0 -or -not $evaluationRetryRuntime) {
    throw "Failed to inspect the pinned official evaluation retry layers"
}
$evaluationRetryRuntime = $evaluationRetryRuntime | ConvertFrom-Json
if (
    $evaluationRetryRuntime.openai_sdk_version -ne "2.16.0" -or
    [int]$evaluationRetryRuntime.openai_sdk_default_max_retries -ne 2 -or
    [int]$evaluationRetryRuntime.benchmark_tenacity_max_attempts -ne 5
) {
    throw "Official simulator/judge retry runtime differs from the audited contract"
}
$env:STATE_BENCH_AGENT_MAX_TOKENS = "4096"
$env:STATE_BENCH_AGENT_TIMEOUT_SECONDS = "120"
$env:STATE_BENCH_AGENT_MAX_RETRIES = "0"

if ($env:STATE_BENCH_AGENT_MODEL -cne "gpt-5.4") {
    throw "Official stages require STATE_BENCH_AGENT_MODEL=gpt-5.4"
}
$evaluationDeployments = Get-EvaluationDeploymentMap
if ($RunStart -ne 1) {
    throw "The locked protocol requires -RunStart 1"
}
if ($Workers -lt 1 -or $Workers -gt 3) {
    throw "Workers must be between 1 and 3; use 2 until the relay is proven stable."
}
if ($Stage -eq "official750" -and $Workers -ne 2) {
    throw "official750 is locked to exactly 2 workers; automatic ramp-up is not part of this protocol."
}
if ($Stage -eq "official750" -and ($Arm -ne "candidate" -or $RouterStage -ne "C")) {
    throw "official750 is valid only for -Arm candidate -RouterStage C"
}
$officialDomains = @("shopping_assistant", "travel", "customer_support")
$requestedDomains = @($Domains | Sort-Object -Unique)
if (
    $Stage -eq "official750" -and
    ($requestedDomains.Count -ne 3 -or @($officialDomains | Where-Object { $_ -notin $requestedDomains }).Count -ne 0)
) {
    throw "official750 requires all three official domains"
}
if (-not $StartRelay) {
    throw "All protocol stages require -StartRelay so the 45-RPM/burst-5 transport is attributable."
}

$artifactDirectory = Join-Path $repoRoot "artifacts\statebench_cross_domain_pwm\memory"
if ($Stage -in @("dev", "lockbox")) {
    $artifactKind = "optimizer80"
    $memoryPath = Join-Path $artifactDirectory "process_workflows_optimizer80.json"
    $routerPath = Join-Path $artifactDirectory "workflow_router_v2_optimizer80.json"
} else {
    $artifactKind = "full100"
    $memoryPath = Join-Path $artifactDirectory "process_workflows.json"
    $routerPath = Join-Path $artifactDirectory "workflow_router_v2.json"
}
if (-not (Test-Path -LiteralPath $memoryPath -PathType Leaf)) {
    throw "Missing required $artifactKind memory artifact (fail-closed): $memoryPath"
}
if (-not (Test-Path -LiteralPath $routerPath -PathType Leaf)) {
    throw "Missing required $artifactKind router artifact (fail-closed): $routerPath"
}

$expectedStateBenchCommit = "5644b1838d96bc4483da29642d058ecaa6f80f7f"
$stateBenchCommit = (& git -C $stateBench rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $stateBenchCommit -ne $expectedStateBenchCommit) {
    throw "STATE-Bench must be pinned at $expectedStateBenchCommit; found $stateBenchCommit"
}
$stateBenchTrackedChanges = @(& git -C $stateBench status --porcelain=v1 --untracked-files=no)
if ($LASTEXITCODE -ne 0) { throw "Failed to inspect the STATE-Bench tracked worktree" }
if ($stateBenchTrackedChanges.Count -ne 0) {
    throw "STATE-Bench tracked files must be clean; untracked user files are allowed: $($stateBenchTrackedChanges -join '; ')"
}
$stateBenchTrackedTreeClean = $true
$repositoryCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repositoryCommit) {
    throw "Failed to resolve the selective-PWM repository commit"
}
$repositoryTrackedChanges = @(& git -C $repoRoot status --porcelain=v1 --untracked-files=no)
if ($LASTEXITCODE -ne 0) { throw "Failed to inspect the selective-PWM tracked worktree" }
if ($repositoryTrackedChanges.Count -ne 0) {
    throw "Selective-PWM tracked files must be clean before evaluation: $($repositoryTrackedChanges -join '; ')"
}
$repositoryTrackedTreeClean = $true
$implementationPaths = [ordered]@{
    runner = $PSCommandPath
    risk_aware_agent = Join-Path $repoRoot "agents\risk_aware_process_workflow_memory_agent.py"
    parent_agent = Join-Path $repoRoot "agents\process_workflow_memory_agent.py"
    actor_agent = Join-Path $repoRoot "agents\opencode_agent.py"
    agent_client = Join-Path $repoRoot "clients\opencode_client.py"
    relay = Join-Path $repoRoot "tools\eval_shim.py"
    gate_evaluator = Join-Path $repoRoot "scripts\evaluate_gate.py"
    official_validator = Join-Path $repoRoot "scripts\validate_official_submission.py"
    billing_reconciler = Join-Path $repoRoot "scripts\reconcile_novacode_billing.py"
    v1_builder = Join-Path $repoRoot "scripts\build_process_workflows.py"
    router_builder = Join-Path $repoRoot "scripts\build_workflow_router_v2.py"
    optimizer_builder = Join-Path $repoRoot "scripts\build_optimizer80_artifacts.ps1"
    gate_config = Join-Path $repoRoot "configs\evaluation_gates.json"
    split_manifest = Join-Path $repoRoot "configs\workflow_router_dev_ids.json"
    router_schema = Join-Path $repoRoot "docs\workflow_router_v2.schema.json"
    billing_schema = Join-Path $repoRoot "docs\novacode_billing_evidence.schema.json"
    resume_protocol = Join-Path $repoRoot "scripts\resume_protocol.py"
    artifact_preflight = Join-Path $repoRoot "scripts\preflight_training_artifacts.py"
}
$implementationHashes = [ordered]@{}
foreach ($entry in $implementationPaths.GetEnumerator()) {
    $relative = [IO.Path]::GetRelativePath($repoRoot, $entry.Value).Replace("\", "/")
    & git -C $repoRoot ls-files --error-unmatch -- $relative *> $null
    if ($LASTEXITCODE -ne 0) { throw "Runtime/config file is not tracked by Git: $relative" }
    $workingBlob = (& git -C $repoRoot hash-object -- $relative).Trim()
    $commitBlob = (& git -C $repoRoot rev-parse "$repositoryCommit`:$relative").Trim()
    if ($LASTEXITCODE -ne 0 -or $workingBlob -ne $commitBlob) {
        throw "Runtime/config file does not match the declared commit: $relative"
    }
    $implementationHashes[$entry.Key] = Get-FileSha256 $entry.Value
}
foreach ($artifactPath in ($memoryPath, $routerPath)) {
    $relative = [IO.Path]::GetRelativePath($repoRoot, $artifactPath).Replace("\", "/")
    & git -C $repoRoot ls-files --error-unmatch -- $relative *> $null
    if ($LASTEXITCODE -ne 0) { throw "Evaluation artifact is not tracked by Git: $relative" }
    $workingBlob = (& git -C $repoRoot hash-object -- $relative).Trim()
    $commitBlob = (& git -C $repoRoot rev-parse "$repositoryCommit`:$relative").Trim()
    if ($LASTEXITCODE -ne 0 -or $workingBlob -ne $commitBlob) {
        throw "Evaluation artifact does not match the declared commit: $relative"
    }
}
& $script:stateBenchPython (Join-Path $repoRoot "scripts\preflight_training_artifacts.py") `
    --kind $artifactKind `
    --memory $memoryPath `
    --router $routerPath `
    --state-bench-root $stateBench `
    --repository-root $repoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Training-artifact provenance preflight failed closed before relay/API startup"
}
$stateBenchVersion = Get-StateBenchVersion $stateBench
if ($stateBenchVersion -ne "0.8.1") { throw "STATE-Bench version must be 0.8.1; found $stateBenchVersion" }
$protocolPath = Join-Path $stateBench "state_bench\configs\eval_protocols\gpt54.json"
$protocol = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
if (
    $protocol.split_version -ne "train_test" -or
    $protocol.split -ne "test" -or
    [int]$protocol.num_runs -ne 5 -or
    $protocol.simulator.model -ne "gpt-5.4" -or
    $protocol.judge.model -ne "gpt-5.4" -or
    $protocol.judge.reasoning_effort -ne "high" -or
    $protocol.official_model -ne "gpt-5.4" -or
    @($protocol.domains).Count -ne 3 -or
    @($officialDomains | Where-Object { $_ -notin @($protocol.domains) }).Count -ne 0
) {
    throw "STATE-Bench gpt54 protocol file does not match the locked official contract"
}

$env:STATE_BENCH_MEMORY_PATH = $memoryPath
$env:STATE_BENCH_WORKFLOW_ROUTER_PATH = $routerPath
$env:STATE_BENCH_WORKFLOW_ROUTER_MODE = "enforce"
$env:STATE_BENCH_WORKFLOW_ROUTER_STAGE = $RouterStage
$env:STATE_BENCH_MEMORY_MODE = "hybrid"
$env:NO_PROXY = (($env:NO_PROXY, "127.0.0.1", "localhost") -join ",").Trim(",")
$env:no_proxy = $env:NO_PROXY

$armDirectoryName = if ($Arm -eq "candidate") { "$Arm-$RouterStage" } else { $Arm }
$armRootPreflight = Join-Path $repoRoot "$OutputRoot\$Stage\$armDirectoryName"
if ($Resume) {
    if (-not (Test-Path -LiteralPath $armRootPreflight -PathType Container)) {
        throw "-Resume requires the existing arm output root: $armRootPreflight"
    }
    foreach ($domain in $Domains) {
        $manifestPath = Join-Path $armRootPreflight "$domain\run_manifest.json"
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            throw "-Resume requires the immutable run manifest: $manifestPath"
        }
    }
} else {
    if (Test-Path -LiteralPath $armRootPreflight -PathType Container) {
        $armExisting = Get-ChildItem -LiteralPath $armRootPreflight -Recurse -File -ErrorAction SilentlyContinue
        if ($armExisting) { throw "Arm output is not empty; fresh protocol runs require a new directory: $armRootPreflight" }
    }
}

$relay = $null
$relayPort = 8765
$relaySessionId = [Guid]::NewGuid().ToString("N")
$armRoot = Join-Path $repoRoot "$OutputRoot\$Stage\$armDirectoryName"
$transportDirectory = Join-Path $armRoot "_transport"
New-Item -ItemType Directory -Force -Path $transportDirectory | Out-Null
$relayLog = Join-Path $transportDirectory "relay-$relaySessionId.log"
$relayLedger = Join-Path $transportDirectory "relay-$relaySessionId.jsonl"
$upstreamUri = [Uri]$env:STATE_BENCH_AGENT_BASE_URL
if (-not $upstreamUri.IsAbsoluteUri -or $upstreamUri.Scheme -notin @("http", "https")) {
    throw "STATE_BENCH_AGENT_BASE_URL must be an absolute HTTP(S) provider URL"
}
$defaultPort = ($upstreamUri.Scheme -eq "https" -and $upstreamUri.Port -eq 443) -or
    ($upstreamUri.Scheme -eq "http" -and $upstreamUri.Port -eq 80)
$upstreamOrigin = "$($upstreamUri.Scheme.ToLowerInvariant())://$($upstreamUri.Host.ToLowerInvariant())"
if (-not $defaultPort) { $upstreamOrigin += ":$($upstreamUri.Port)" }
$upstreamOriginSha256 = Get-TextSha256 $upstreamOrigin
$transportEvidence = [ordered]@{
    provider = "novacode"
    upstream_origin_sha256 = $upstreamOriginSha256
    relay_session_id = $relaySessionId
    relay_sha256 = $implementationHashes.relay
    rpm = 45
    burst = 5
    burst_window_seconds = 1.0
    attempts = 5
    only_transport_retries = $true
    ledger_relative_path = "_transport/relay-$relaySessionId.jsonl"
    log_relative_path = "_transport/relay-$relaySessionId.log"
}
if ($StartRelay) {
    if (Get-NetTCPConnection -LocalPort $relayPort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $relayPort must be free so this run can start its own attributable relay"
    }
    $upstream = $env:STATE_BENCH_AGENT_BASE_URL.TrimEnd("/")
    if ($upstream -match '^http://127\.0\.0\.1:8765') {
        throw "Env file already points the agent at the relay; SHIM_UPSTREAM must be the real provider URL."
    }
    $env:SHIM_UPSTREAM = $upstream
    $env:SHIM_PORT = [string]$relayPort
    $env:SHIM_RPM = "45"
    $env:SHIM_BURST = "5"
    $env:SHIM_BURST_WINDOW = "1.0"
    $env:SHIM_ATTEMPTS = "5"
    $env:SHIM_LEDGER_PATH = $relayLedger
    $relay = Start-Process -FilePath $script:stateBenchPython `
        -ArgumentList @((Join-Path $repoRoot "tools\eval_shim.py")) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $relayLog `
        -RedirectStandardError ($relayLog + ".err") `
        -WindowStyle Hidden `
        -PassThru
    Wait-Relay -Port $relayPort
    $env:STATE_BENCH_AGENT_BASE_URL = "http://127.0.0.1:$relayPort/v1"
    $env:STATE_BENCH_EVAL_ENDPOINT = "http://127.0.0.1:$relayPort"
}

try {
    $router = if (Test-Path -LiteralPath $routerPath) {
        Get-Content -LiteralPath $routerPath -Raw | ConvertFrom-Json
    } else { $null }
    $agentClass = if ($Arm -eq "candidate") {
        "RiskAwareProcessWorkflowMemoryAgent"
    } else {
        "ProcessWorkflowMemoryAgent"
    }
    $runs = if ($Stage -in @("lockbox", "official750")) { 2 } else { 1 }
    if ($Stage -eq "official750") { $runs = 5 }
    $resumeHelper = Join-Path $repoRoot "scripts\resume_protocol.py"
    $executionIncomplete = $false

    foreach ($domain in $Domains) {
        if ($domain -notin @("shopping_assistant", "travel", "customer_support")) {
            throw "Unknown domain: $domain"
        }
        $out = Join-Path $repoRoot "$OutputRoot\$Stage\$armDirectoryName\$domain"
        New-Item -ItemType Directory -Force -Path $out | Out-Null

        $splitManifestPath = Join-Path $stateBench "state_bench\domains\$domain\splits\$($protocol.split_version).json"
        $splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw | ConvertFrom-Json
        if ($Stage -in @("paired150", "official750")) {
            $ids = @($splitManifest.splits.test | ForEach-Object { [string]$_ })
            $taskSource = "state_bench_official_test_split"
            $taskMode = "split"
        } else {
            if ($null -eq $router -or $null -eq $router.splits.$domain) {
                throw "Router artifact has no split manifest for $domain"
            }
            $ids = if ($Stage -eq "dev") {
                @($router.splits.$domain.dev | ForEach-Object { [string]$_ })
            } else {
                @($router.splits.$domain.lockbox | ForEach-Object { [string]$_ })
            }
            $taskSource = "optimizer80_router_$Stage"
            $taskMode = "explicit"
        }
        $expectedTaskCount = if ($Stage -in @("dev", "lockbox")) { 10 } else { 50 }
        if ($ids.Count -ne $expectedTaskCount -or @($ids | Sort-Object -Unique).Count -ne $expectedTaskCount) {
            throw "$Stage task selection for $domain must contain exactly $expectedTaskCount unique IDs"
        }
        $runManifestPath = Join-Path $out "run_manifest.json"
        $manifestTransportEvidence = $transportEvidence
        if ($Resume) {
            $storedManifest = Get-Content -LiteralPath $runManifestPath -Raw | ConvertFrom-Json
            $manifestTransportEvidence = $storedManifest.transport
        }
        $manifest = New-RunManifest `
            -Domain $domain `
            -TaskIds $ids `
            -TaskSource $taskSource `
            -TaskMode $taskMode `
            -AgentClass $agentClass `
            -Runs $runs `
            -MemoryPath $memoryPath `
            -RouterPath $routerPath `
            -ArtifactKind $artifactKind `
            -RunnerPath $PSCommandPath `
            -RepositoryCommit $repositoryCommit `
            -RepositoryTrackedTreeClean $repositoryTrackedTreeClean `
            -ImplementationHashes $implementationHashes `
            -TransportEvidence $manifestTransportEvidence `
            -StateBenchCommit $stateBenchCommit `
            -StateBenchVersion $stateBenchVersion `
            -ProtocolPath $protocolPath `
            -Protocol $protocol `
            -SplitManifestPath $splitManifestPath `
            -EvaluationDeployments $evaluationDeployments `
            -StateBenchTrackedTreeClean $stateBenchTrackedTreeClean
        Write-RunManifestExclusive `
            -Path $runManifestPath `
            -Expected $manifest `
            -IsResume:$Resume
        $taskIdsPath = New-TaskIdFile -TaskIds $ids
        $domainComplete = $true
        try {
            for ($runIndex = $RunStart; $runIndex -lt ($RunStart + $runs); $runIndex++) {
                $runDirectory = Join-Path $out "run$runIndex"
                if (-not $Resume) {
                    if ($Stage -eq "official750") {
                        # The official protocol is one literal five-run batch per
                        # domain.  Later loop members are projections of this
                        # single invocation, not additional run_batch calls.
                        if ($runIndex -ne 1) { continue }
                        $batchRunIndices = @(1, 2, 3, 4, 5)
                        $preSnapshotPaths = [ordered]@{}
                        $preSnapshotsInputPath = Join-Path `
                            ([IO.Path]::GetTempPath()) `
                            "selective-pwm-batch-snapshots-$([Guid]::NewGuid().ToString('N')).json"
                        try {
                            foreach ($batchRunIndex in $batchRunIndices) {
                                $batchRunDirectory = Join-Path $out "run$batchRunIndex"
                                $snapshotPath = New-RunSnapshot `
                                    -ResumeHelper $resumeHelper `
                                    -RunDirectory $batchRunDirectory `
                                    -TaskIdsPath $taskIdsPath
                                $preSnapshotPaths[[string]$batchRunIndex] = $snapshotPath
                            }
                            Write-JsonExclusive `
                                -Path $preSnapshotsInputPath `
                                -Value ([ordered]@{
                                    run_indices = $batchRunIndices
                                    pre_snapshot_paths = $preSnapshotPaths
                                })
                            $batchLog = New-AuditLogPath `
                                -ArmRoot $armRoot `
                                -Domain $domain `
                                -RunIndex 1 `
                                -Mode "fresh-official-runs1-5"
                            $relayStart = Get-FileLengthOrZero $relayLedger
                            $batchArgs = @(
                                "run", "--project", $stateBench, "python", "-m", "state_bench.scripts.run_batch",
                                "--domain", $domain,
                                "--output-dir", $out,
                                "--split", "test",
                                "--num-runs", "5",
                                "--num-runs-idx-start", "1",
                                "--num-workers", [string]$Workers,
                                "--retry-attempts", "1",
                                "--agent-class", $agentClass,
                                "--agent-client-class", "OpenCodeLLMClient",
                                "--retrieve-learnings-top-k", "3",
                                "--agent-model-name", "gpt-5.4",
                                "--score-reasoning-effort", "high"
                            )
                            Write-Host (
                                "Running one official fresh batch: $Arm / router=$RouterStage / $domain / " +
                                "runs1..5 / $($ids.Count) task(s) per run, workers=$Workers"
                            )
                            $batchExit = Invoke-CapturedNative `
                                -Executable "uv" `
                                -Arguments $batchArgs `
                                -LogPath $batchLog
                            Write-AuditOfficialBatchSession `
                                -ResumeHelper $resumeHelper `
                                -ArmRoot $armRoot `
                                -Domain $domain `
                                -RunManifest $runManifestPath `
                                -AllTaskIdsPath $taskIdsPath `
                                -PreSnapshotsJson $preSnapshotsInputPath `
                                -LogPath $batchLog `
                                -RelayLedger $relayLedger `
                                -RelayStartOffset $relayStart `
                                -ProcessExitCode $batchExit
                        } finally {
                            foreach ($snapshotPath in @($preSnapshotPaths.Values)) {
                                Remove-TemporaryFile ([string]$snapshotPath)
                            }
                            Remove-TemporaryFile $preSnapshotsInputPath
                        }

                        foreach ($batchRunIndex in $batchRunIndices) {
                            $postPlan = Get-ResumePlan `
                                -ResumeHelper $resumeHelper `
                                -ArmRoot $armRoot `
                                -Domain $domain `
                                -RunIndex $batchRunIndex `
                                -RunManifest $runManifestPath `
                                -TaskIdsPath $taskIdsPath
                            if (
                                $batchExit -ne 0 -or
                                @($postPlan.agent_task_ids).Count -ne 0 -or
                                @($postPlan.score_task_ids).Count -ne 0 -or
                                @($postPlan.rejected_task_ids).Count -ne 0
                            ) {
                                $executionIncomplete = $true
                                $domainComplete = $false
                                Write-Warning (
                                    "$domain run$batchRunIndex is incomplete. Its shared fresh-batch record " +
                                    "and per-run projection determine whether a later -Resume is allowed."
                                )
                            }
                        }
                        continue
                    }
                    $preSnapshot = New-RunSnapshot `
                        -ResumeHelper $resumeHelper `
                        -RunDirectory $runDirectory `
                        -TaskIdsPath $taskIdsPath
                    $batchLog = New-AuditLogPath `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -Mode "fresh"
                    $relayStart = Get-FileLengthOrZero $relayLedger
                    $batchArgs = @(
                        "run", "--project", $stateBench, "python", "-m", "state_bench.scripts.run_batch",
                        "--domain", $domain,
                        "--output-dir", $out,
                        "--num-runs", "1",
                        "--num-runs-idx-start", [string]$runIndex,
                        "--num-workers", [string]$Workers,
                        "--retry-attempts", "1",
                        "--agent-class", $agentClass,
                        "--agent-client-class", "OpenCodeLLMClient",
                        "--retrieve-learnings-top-k", "3",
                        "--agent-model-name", "gpt-5.4",
                        "--score-reasoning-effort", "high"
                    )
                    if ($Stage -in @("paired150", "official750")) {
                        $batchArgs += @("--split", "test")
                    } else {
                        $batchArgs += @("--tasks", (@($ids) -join ","))
                    }
                    Write-Host (
                        "Running fresh $Stage / $Arm / router=$RouterStage / $domain / " +
                        "run$runIndex / $($ids.Count) task(s), workers=$Workers"
                    )
                    $batchExit = Invoke-CapturedNative `
                        -Executable "uv" `
                        -Arguments $batchArgs `
                        -LogPath $batchLog
                    Write-AuditSession `
                        -ResumeHelper $resumeHelper `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -RunManifest $runManifestPath `
                        -Mode "fresh" `
                        -AllTaskIdsPath $taskIdsPath `
                        -TargetTaskIdsPath $taskIdsPath `
                        -PreSnapshot $preSnapshot `
                        -LogPath $batchLog `
                        -RelayLedger $relayLedger `
                        -RelayStartOffset $relayStart `
                        -ProcessExitCode $batchExit
                    Remove-TemporaryFile $preSnapshot

                    $postPlan = Get-ResumePlan `
                        -ResumeHelper $resumeHelper `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -RunManifest $runManifestPath `
                        -TaskIdsPath $taskIdsPath
                    if (
                        $batchExit -ne 0 -or
                        @($postPlan.agent_task_ids).Count -ne 0 -or
                        @($postPlan.score_task_ids).Count -ne 0 -or
                        @($postPlan.rejected_task_ids).Count -ne 0
                    ) {
                        $executionIncomplete = $true
                        $domainComplete = $false
                        Write-Warning (
                            "$domain run$runIndex is incomplete. Its immutable session record determines " +
                            "whether a later -Resume may rerun a missing agent trajectory or score an existing raw trajectory."
                        )
                    }
                    continue
                }

                $resumePlan = Get-ResumePlan `
                    -ResumeHelper $resumeHelper `
                    -ArmRoot $armRoot `
                    -Domain $domain `
                    -RunIndex $runIndex `
                    -RunManifest $runManifestPath `
                    -TaskIdsPath $taskIdsPath
                $rejectedIds = @($resumePlan.rejected_task_ids | ForEach-Object { [string]$_ })
                if ($rejectedIds.Count -gt 0) {
                    $preSnapshot = New-RunSnapshot $resumeHelper $runDirectory $taskIdsPath
                    $batchLog = New-AuditLogPath $armRoot $domain $runIndex "resume-rejected"
                    Write-AuditNoteExclusive `
                        -Path $batchLog `
                        -Message "Resume rejected: latest immutable task evidence is absent or non-transport."
                    $relayStart = Get-FileLengthOrZero $relayLedger
                    Write-AuditSession `
                        -ResumeHelper $resumeHelper `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -RunManifest $runManifestPath `
                        -Mode "resume_rejected" `
                        -AllTaskIdsPath $taskIdsPath `
                        -TargetTaskIdsPath "" `
                        -PreSnapshot $preSnapshot `
                        -LogPath $batchLog `
                        -RelayLedger $relayLedger `
                        -RelayStartOffset $relayStart `
                        -ProcessExitCode 2
                    Remove-TemporaryFile $preSnapshot
                    $executionIncomplete = $true
                    $domainComplete = $false
                    Write-Warning "$domain run$runIndex resume rejected fail-closed; no trajectory was executed or rescored."
                    continue
                }

                $agentTaskIds = @($resumePlan.agent_task_ids | ForEach-Object { [string]$_ })
                $scoreTaskIds = @($resumePlan.score_task_ids | ForEach-Object { [string]$_ })
                $performedAction = $false
                if ($agentTaskIds.Count -gt 0) {
                    $performedAction = $true
                    foreach ($agentTaskId in $agentTaskIds) {
                        $agentIdsPath = New-TaskIdFile -TaskIds @($agentTaskId)
                        $preSnapshot = New-RunSnapshot $resumeHelper $runDirectory $taskIdsPath
                        $batchLog = New-AuditLogPath $armRoot $domain $runIndex "resume-agent"
                        $relayStart = Get-FileLengthOrZero $relayLedger
                        $batchArgs = @(
                            "run", "--project", $stateBench, "python", "-m", "state_bench.scripts.run_batch",
                            "--domain", $domain,
                            "--output-dir", $out,
                            "--num-runs", "1",
                            "--num-runs-idx-start", [string]$runIndex,
                            "--num-workers", "1",
                            "--retry-attempts", "1",
                            "--agent-class", $agentClass,
                            "--agent-client-class", "OpenCodeLLMClient",
                            "--retrieve-learnings-top-k", "3",
                            "--agent-model-name", "gpt-5.4",
                            "--score-reasoning-effort", "high",
                            "--tasks", $agentTaskId
                        )
                        Write-Host "Resuming one transport-proved missing trajectory: $domain run$runIndex / $agentTaskId"
                        $batchExit = Invoke-CapturedNative "uv" $batchArgs $batchLog
                        Write-AuditSession `
                            -ResumeHelper $resumeHelper `
                            -ArmRoot $armRoot `
                            -Domain $domain `
                            -RunIndex $runIndex `
                            -RunManifest $runManifestPath `
                            -Mode "resume_agent" `
                            -AllTaskIdsPath $taskIdsPath `
                            -TargetTaskIdsPath $agentIdsPath `
                            -PreSnapshot $preSnapshot `
                            -LogPath $batchLog `
                            -RelayLedger $relayLedger `
                            -RelayStartOffset $relayStart `
                            -ProcessExitCode $batchExit
                        Remove-TemporaryFile $preSnapshot
                        Remove-TemporaryFile $agentIdsPath
                    }
                }

                foreach ($taskId in $scoreTaskIds) {
                    $performedAction = $true
                    $singleTaskPath = New-TaskIdFile -TaskIds @($taskId)
                    $preSnapshot = New-RunSnapshot $resumeHelper $runDirectory $taskIdsPath
                    $preState = Get-Content -LiteralPath $preSnapshot -Raw | ConvertFrom-Json
                    $taskProperty = $preState.tasks.PSObject.Properties[$taskId]
                    if ($null -eq $taskProperty -or $taskProperty.Value.state -ne "unscored") {
                        throw "Score-only resume source is not an unscored trajectory: $taskId"
                    }
                    $sourceHash = [string]$taskProperty.Value.sha256
                    $scoreSessionId = [Guid]::NewGuid().ToString("N")
                    $scoreRoot = Join-Path $armRoot "_resume_tmp\$scoreSessionId\$domain"
                    $stagedPath = Join-Path $scoreRoot "run$runIndex\$taskId.json"
                    $sourcePath = Join-Path $runDirectory "$taskId.json"
                    & $script:stateBenchPython $resumeHelper stage-score `
                        --source $sourcePath `
                        --destination $stagedPath `
                        --task-id $taskId `
                        --expected-source-sha256 $sourceHash
                    if ($LASTEXITCODE -ne 0) { throw "Failed to stage isolated score-only retry for $taskId" }

                    $batchLog = New-AuditLogPath $armRoot $domain $runIndex "resume-score"
                    $relayStart = Get-FileLengthOrZero $relayLedger
                    $scoreArgs = @(
                        "run", "--project", $stateBench, "python", "-m", "state_bench.scripts.score",
                        "--domain", $domain,
                        "--results-dir", $scoreRoot,
                        "--num-runs", [string]$runs,
                        "--num-runs-idx-start", [string]$RunStart,
                        "--num-workers", "1",
                        "--reasoning-effort", "high",
                        "--split", $(if ($Stage -in @("paired150", "official750")) { "test" } else { "all" })
                    )
                    Write-Host "Score-only resume in isolated root: $domain run$runIndex / $taskId"
                    $scoreExit = Invoke-CapturedNative "uv" $scoreArgs $batchLog
                    if ($scoreExit -eq 0) {
                        $promotion = & $script:stateBenchPython $resumeHelper promote-score `
                            --staged $stagedPath `
                            --destination $sourcePath `
                            --task-id $taskId `
                            --expected-destination-sha256 $sourceHash
                        if ($LASTEXITCODE -ne 0) { throw "Score-only promotion validation failed for $taskId" }
                    }
                    Write-AuditSession `
                        -ResumeHelper $resumeHelper `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -RunManifest $runManifestPath `
                        -Mode "resume_score" `
                        -AllTaskIdsPath $taskIdsPath `
                        -TargetTaskIdsPath $singleTaskPath `
                        -PreSnapshot $preSnapshot `
                        -LogPath $batchLog `
                        -RelayLedger $relayLedger `
                        -RelayStartOffset $relayStart `
                        -ProcessExitCode $scoreExit
                    Remove-TemporaryFile $preSnapshot
                    Remove-TemporaryFile $singleTaskPath
                    $resolvedScoreRoot = [IO.Path]::GetFullPath($scoreRoot)
                    $resolvedArmRoot = [IO.Path]::GetFullPath($armRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
                    if (-not $resolvedScoreRoot.StartsWith($resolvedArmRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                        throw "Refusing to remove score staging outside the arm root"
                    }
                    if (Test-Path -LiteralPath $resolvedScoreRoot -PathType Container) {
                        Remove-Item -LiteralPath $resolvedScoreRoot -Recurse -Force
                    }
                }

                if (-not $performedAction) {
                    $preSnapshot = New-RunSnapshot $resumeHelper $runDirectory $taskIdsPath
                    $batchLog = New-AuditLogPath $armRoot $domain $runIndex "resume-noop"
                    Write-AuditNoteExclusive `
                        -Path $batchLog `
                        -Message "Resume no-op: every expected trajectory is already fully scored and immutable."
                    $relayStart = Get-FileLengthOrZero $relayLedger
                    Write-AuditSession `
                        -ResumeHelper $resumeHelper `
                        -ArmRoot $armRoot `
                        -Domain $domain `
                        -RunIndex $runIndex `
                        -RunManifest $runManifestPath `
                        -Mode "resume_noop" `
                        -AllTaskIdsPath $taskIdsPath `
                        -TargetTaskIdsPath "" `
                        -PreSnapshot $preSnapshot `
                        -LogPath $batchLog `
                        -RelayLedger $relayLedger `
                        -RelayStartOffset $relayStart `
                        -ProcessExitCode 0
                    Remove-TemporaryFile $preSnapshot
                }

                $postPlan = Get-ResumePlan $resumeHelper $armRoot $domain $runIndex $runManifestPath $taskIdsPath
                if (
                    @($postPlan.agent_task_ids).Count -ne 0 -or
                    @($postPlan.score_task_ids).Count -ne 0 -or
                    @($postPlan.rejected_task_ids).Count -ne 0
                ) {
                    $executionIncomplete = $true
                    $domainComplete = $false
                    Write-Warning (
                        "$domain run$runIndex remains incomplete. A repeated transport failure can be retried " +
                        "only by a new -Resume invocation using the session just written."
                    )
                }
            }
            if ($domainComplete -and $Stage -in @("paired150", "official750")) {
                $metricArgs = @(
                    "run", "--project", $stateBench, "python", "-m", "state_bench.scripts.compute_metrics",
                    "--domain", $domain,
                    "--results-dir", $out,
                    "--num-runs", [string]$runs,
                    "--num-runs-idx-start", [string]$RunStart,
                    "--split", "test",
                    "--save-filepath", (Join-Path $out "metrics.json")
                )
                & uv @metricArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Strict compute_metrics failed with code $LASTEXITCODE (missing runs are never ignored)"
                }
            }
        } finally {
            Remove-TemporaryFile $taskIdsPath
        }
    }

    if ($null -ne $relay -and -not $relay.HasExited) {
        Stop-Process -Id $relay.Id
        Wait-Process -Id $relay.Id -ErrorAction SilentlyContinue
        $relay = $null
    }
    if (-not (Test-Path -LiteralPath $relayLedger -PathType Leaf)) {
        throw "The append-only relay usage ledger is missing: $relayLedger"
    }
    if ($executionIncomplete) {
        throw "Evaluation remains incomplete; strict metrics and promotion gates were not accepted. Inspect only the sanitized session records before using -Resume."
    }

    $allDomains = @("shopping_assistant", "travel", "customer_support")
    $requested = @($Domains | Sort-Object -Unique)
    $completeDomainSet = $requested.Count -eq 3 -and @($allDomains | Where-Object { $_ -notin $requested }).Count -eq 0
    if ($completeDomainSet -and $Arm -eq "candidate") {
        $stageRoot = Join-Path $repoRoot "$OutputRoot\$Stage"
        $candidateRoot = Join-Path $stageRoot "candidate-$RouterStage"
        if ($Stage -eq "official750") {
            & $script:stateBenchPython (Join-Path $repoRoot "scripts\evaluate_gate.py") `
                --gate official750 `
                --candidate $candidateRoot `
                --state-bench-root $stateBench `
                --memory $memoryPath `
                --router $routerPath
            if ($LASTEXITCODE -ne 0) { throw "official750 score/execution gate failed" }
            $billingEvidence = Join-Path $candidateRoot "billing_evidence.json"
            if (Test-Path -LiteralPath $billingEvidence -PathType Leaf) {
                & $script:stateBenchPython (Join-Path $repoRoot "scripts\validate_official_submission.py") `
                    --candidate $candidateRoot `
                    --state-bench-root $stateBench `
                    --memory $memoryPath `
                    --router $routerPath `
                    --runner $PSCommandPath `
                    --billing-evidence $billingEvidence
                if ($LASTEXITCODE -ne 0) { throw "official750 formal validation failed" }
            } else {
                Write-Warning (
                    "Official scores passed, but the result is not formally claimable until " +
                    "NovaCode billing_evidence.json is reconciled and validate_official_submission.py passes."
                )
            }
        } else {
            $baselineRoot = Join-Path $stageRoot "baseline"
            if (Test-Path -LiteralPath $baselineRoot -PathType Container) {
                & $script:stateBenchPython (Join-Path $repoRoot "scripts\evaluate_gate.py") `
                    --gate $Stage `
                    --baseline $baselineRoot `
                    --candidate $candidateRoot `
                    --state-bench-root $stateBench `
                    --memory $memoryPath `
                    --router $routerPath
                if ($LASTEXITCODE -ne 0) { throw "$Stage promotion gate failed for router stage $RouterStage" }
            } else {
                Write-Warning "Baseline root is absent; gate pending: $baselineRoot"
            }
        }
    } elseif (-not $completeDomainSet) {
        Write-Warning "Partial domain run completed; cross-domain gate remains pending."
    }
} finally {
    if ($null -ne $relay -and -not $relay.HasExited) {
        Stop-Process -Id $relay.Id
    }
    Remove-Item Env:\SHIM_LEDGER_PATH -ErrorAction SilentlyContinue
}
