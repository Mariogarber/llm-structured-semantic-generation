param(
    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [int]$SampleOffset = 100,
    [int]$MaxTrainSamples = 200,
    [int]$NumCandidates = 6,
    [string]$Temperatures = "0.45,0.65,0.85,1.0,1.1,1.2",
    [double]$TopP = 0.95,
    [int]$MaxNewTokens = 512,
    [int]$BatchSize = 1,
    [int]$IntervalMinutes = 60,
    [int]$MaxGpuTempC = 82,
    [int]$ResumeGpuTempC = 68,
    [int]$ExternalGpuUtilizationThreshold = 10,
    [int]$ExternalGpuMemoryThresholdMiB = 1000
)

$ErrorActionPreference = "Stop"

$ResolvedWorkspace = (Resolve-Path -LiteralPath $Workspace).Path
$RunDir = Join-Path $ResolvedWorkspace "results\dpo_kubernetes_v1\candidate_generation\$RunId"
$LogDir = Join-Path $ResolvedWorkspace "results\dpo_kubernetes_v1\candidate_generation\logs"
$WatchLog = Join-Path $LogDir "$RunId.health-watch.log"
$Python = Join-Path $ResolvedWorkspace ".venv\Scripts\python.exe"
$Script = Join-Path $ResolvedWorkspace "scripts\build_kubernetes_dpo_candidates.py"
$StdoutLog = Join-Path $LogDir "$RunId.health-resume.out.log"
$StderrLog = Join-Path $LogDir "$RunId.health-resume.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("s")
    Add-Content -LiteralPath $WatchLog -Value "$stamp $Message"
}

function Get-TargetProcesses {
    return @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$RunId*" })
}

function Get-RunState {
    $statePath = Join-Path $RunDir "state.json"
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
}

function Get-JsonlLineCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Get-GpuSnapshot {
    $gpuRows = & nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $gpuRows) {
        return [pscustomobject]@{
            Available = $false
            Reason = "nvidia_smi_unavailable"
            MaxTemp = $null
            MaxUtilization = $null
            MaxMemory = $null
            Detail = ""
        }
    }

    $temps = @()
    $utils = @()
    $memories = @()
    foreach ($row in $gpuRows) {
        $parts = $row.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 3) {
            continue
        }
        $temps += [int]$parts[0]
        $utils += [int]$parts[1]
        $memories += [int]$parts[2]
    }

    return [pscustomobject]@{
        Available = $true
        Reason = "ok"
        MaxTemp = (($temps | Measure-Object -Maximum).Maximum)
        MaxUtilization = (($utils | Measure-Object -Maximum).Maximum)
        MaxMemory = (($memories | Measure-Object -Maximum).Maximum)
        Detail = ($gpuRows -join " | ")
    }
}

function Test-ExternalGpuBusy {
    param([object]$Gpu)
    if (-not $Gpu.Available) {
        return $true
    }
    return (
        $Gpu.MaxUtilization -gt $ExternalGpuUtilizationThreshold -or
        $Gpu.MaxMemory -gt $ExternalGpuMemoryThresholdMiB
    )
}

function Stop-TargetProcesses {
    $targets = Get-TargetProcesses
    foreach ($target in $targets) {
        Stop-Process -Id $target.ProcessId -Force
    }
    return $targets.Count
}

function Start-TargetRun {
    $arguments = @(
        $Script,
        "--run-id", $RunId,
        "--sample-offset", "$SampleOffset",
        "--max-train-samples", "$MaxTrainSamples",
        "--num-candidates", "$NumCandidates",
        "--temperatures", $Temperatures,
        "--top-p", "$TopP",
        "--max-new-tokens", "$MaxNewTokens",
        "--batch-size", "$BatchSize"
    )
    return Start-Process `
        -FilePath $Python `
        -ArgumentList $arguments `
        -WorkingDirectory $ResolvedWorkspace `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru
}

Write-WatchLog "watcher_started run=$RunId interval_minutes=$IntervalMinutes max_temp=$MaxGpuTempC resume_temp=$ResumeGpuTempC"

while ($true) {
    $state = Get-RunState
    $candidateCount = Get-JsonlLineCount -Path (Join-Path $RunDir "candidates.jsonl")
    $metricCount = Get-JsonlLineCount -Path (Join-Path $RunDir "candidate_metrics.jsonl")
    $targetProcesses = Get-TargetProcesses
    $gpu = Get-GpuSnapshot

    $processed = if ($state -ne $null) { $state.processed_units } else { "unknown" }
    $total = if ($state -ne $null) { $state.total_units } else { "unknown" }
    $remaining = if ($state -ne $null) { $state.remaining_units } else { "unknown" }
    $status = if ($state -ne $null) { $state.status } else { "missing_state" }

    Write-WatchLog "status=$status processed=$processed total=$total remaining=$remaining candidates=$candidateCount metrics=$metricCount active_processes=$($targetProcesses.Count) gpu_temp=$($gpu.MaxTemp) gpu_util=$($gpu.MaxUtilization) gpu_mem=$($gpu.MaxMemory)"

    if ($status -eq "completed") {
        Write-WatchLog "run_completed; watcher_exiting"
        exit 0
    }

    if ($targetProcesses.Count -gt 0) {
        if ($gpu.Available -and $gpu.MaxTemp -ge $MaxGpuTempC) {
            $stopped = Stop-TargetProcesses
            Write-WatchLog "temperature_limit_exceeded temp=$($gpu.MaxTemp); stopped_processes=$stopped"
        }
    } else {
        if ($gpu.Available -and $gpu.MaxTemp -gt $ResumeGpuTempC) {
            Write-WatchLog "not_resuming_yet temp=$($gpu.MaxTemp) resume_temp=$ResumeGpuTempC"
        } elseif (Test-ExternalGpuBusy -Gpu $gpu) {
            Write-WatchLog "not_resuming_external_gpu_busy reason=$($gpu.Reason) detail=$($gpu.Detail)"
        } else {
            $process = Start-TargetRun
            Write-WatchLog "resumed_run pid=$($process.Id)"
        }
    }

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
