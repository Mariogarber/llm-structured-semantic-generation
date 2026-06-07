param(
    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [int]$SampleOffset = 100,
    [int]$MaxTrainSamples = 100,
    [int]$NumCandidates = 6,
    [string]$Temperatures = "0.45,0.65,0.85,1.0,1.1,1.2",
    [double]$TopP = 0.95,
    [int]$MaxNewTokens = 512,
    [int]$BatchSize = 1,
    [int]$IntervalMinutes = 30,
    [int]$GpuUtilizationThreshold = 10,
    [int]$GpuMemoryThresholdMiB = 1000
)

$ErrorActionPreference = "Stop"

$ResolvedWorkspace = (Resolve-Path -LiteralPath $Workspace).Path
$RunDir = Join-Path $ResolvedWorkspace "results\dpo_kubernetes_v1\candidate_generation\$RunId"
$LogDir = Join-Path $ResolvedWorkspace "results\dpo_kubernetes_v1\candidate_generation\logs"
$WatcherLog = Join-Path $LogDir "$RunId.gpu-watch.log"
$StdoutLog = Join-Path $LogDir "$RunId.out.log"
$StderrLog = Join-Path $LogDir "$RunId.err.log"
$Python = Join-Path $ResolvedWorkspace ".venv\Scripts\python.exe"
$Script = Join-Path $ResolvedWorkspace "scripts\build_kubernetes_dpo_candidates.py"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-WatcherLog {
    param([string]$Message)
    $stamp = (Get-Date).ToString("s")
    Add-Content -LiteralPath $WatcherLog -Value "$stamp $Message"
}

function Test-TargetRunProcess {
    $processes = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$RunId*" }
    return @($processes).Count -gt 0
}

function Test-RunCompleted {
    $statePath = Join-Path $RunDir "state.json"
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $false
    }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    return $state.status -eq "completed"
}

function Get-GpuStatus {
    $gpuRows = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $gpuRows) {
        return [pscustomobject]@{
            Available = $false
            Reason = "nvidia_smi_unavailable"
            Detail = ""
        }
    }

    $busyRows = @()
    foreach ($row in $gpuRows) {
        $parts = $row.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 2) {
            continue
        }
        $util = [int]$parts[0]
        $memory = [int]$parts[1]
        if ($util -gt $GpuUtilizationThreshold -or $memory -gt $GpuMemoryThresholdMiB) {
            $busyRows += "util=${util}%,mem=${memory}MiB"
        }
    }

    $computeProcesses = & nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
    $hasComputeProcesses = $LASTEXITCODE -eq 0 -and $computeProcesses
    if ($hasComputeProcesses) {
        $busyRows += "compute_processes=$($computeProcesses -join ';')"
    }

    if ($busyRows.Count -gt 0) {
        return [pscustomobject]@{
            Available = $false
            Reason = "gpu_busy"
            Detail = ($busyRows -join " | ")
        }
    }

    return [pscustomobject]@{
        Available = $true
        Reason = "gpu_free"
        Detail = ($gpuRows -join " | ")
    }
}

Write-WatcherLog "watcher_started run=$RunId interval_minutes=$IntervalMinutes"

while ($true) {
    if (Test-RunCompleted) {
        Write-WatcherLog "run_already_completed; exiting"
        exit 0
    }

    if (Test-TargetRunProcess) {
        Write-WatcherLog "target_run_already_active; exiting"
        exit 0
    }

    $gpuStatus = Get-GpuStatus
    Write-WatcherLog "gpu_check available=$($gpuStatus.Available) reason=$($gpuStatus.Reason) detail=$($gpuStatus.Detail)"
    if ($gpuStatus.Available) {
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
        $process = Start-Process `
            -FilePath $Python `
            -ArgumentList $arguments `
            -WorkingDirectory $ResolvedWorkspace `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdoutLog `
            -RedirectStandardError $StderrLog `
            -PassThru
        Write-WatcherLog "launched_run pid=$($process.Id) stdout=$StdoutLog stderr=$StderrLog"
        exit 0
    }

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
