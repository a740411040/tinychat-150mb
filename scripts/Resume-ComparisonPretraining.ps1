param(
    [Parameter(Mandatory)][ValidateRange(1,3051)][int]$ExpectedStep,
    [Parameter(Mandatory)][ValidateRange(1,99999999)][long]$ExpectedTokens
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runDirectory = Join-Path $projectRoot 'runs\pretrain_clean_100m_v2'
$checkpoint = Join-Path $runDirectory ('step_{0:D8}.pt' -f $ExpectedStep)
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$metrics = Get-Content -LiteralPath (Join-Path $runDirectory 'metrics.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($metrics.stop_reason -ne 'interrupted' -or
    $metrics.optimizer_state_unsafe -isnot [bool] -or $metrics.optimizer_state_unsafe -or
    $metrics.PSObject.Properties.Name -notcontains 'recovery_error' -or $null -ne $metrics.recovery_error -or
    $metrics.checkpoint_step -ne $ExpectedStep -or $metrics.cumulative_valid_loss_tokens -ne $ExpectedTokens -or
    $metrics.planned_loss_tokens -ne 100000000 -or $metrics.checkpoint -ne $checkpoint) {
    throw 'This resume launcher requires the explicitly expected safe paused checkpoint'
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw 'Missing paused checkpoint' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'reports\comparison_v2\predata_wiki_review_ready.txt'))) {
    throw 'Missing data release approval'
}
$active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($projectRoot) -and $_.CommandLine.Contains('tinychat.train_stage1')
})
if ($active.Count) { throw 'An owned training process is already running' }
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONPATH = $projectRoot
$tag = 'comparison_pretrain_resume_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$stdout = Join-Path $projectRoot "logs\$tag.stdout.log"
$stderr = Join-Path $projectRoot "logs\$tag.stderr.log"
$telemetry = Join-Path $projectRoot "logs\$tag.gpu.csv"
$monitorStdout = Join-Path $projectRoot "logs\$tag.monitor.log"
$monitorStderr = Join-Path $projectRoot "logs\$tag.monitor.err.log"
$launchRecord = Join-Path $runDirectory "$tag.launcher.json"
foreach ($path in @($stdout, $stderr, $telemetry, $monitorStdout, $monitorStderr, $launchRecord)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing resume output: $path" }
}
$trainingArguments = @('-X', 'utf8', '-u', '-m', 'tinychat.train_stage1', '--stage', 'pretrain',
    '--data-dir', (Join-Path $projectRoot 'data\pretrain_clean_v2_wiki'), '--run-dir', $runDirectory,
    '--config', (Join-Path $projectRoot 'configs\pilot.json'), '--micro-batch', '8', '--grad-accum', '8',
    '--checkpoint-every', '250', '--eval-every', '250', '--device', 'cuda', '--seed', '42',
    '--target-tokens', '100000000', '--resume', $checkpoint)
$jobProcess = Start-Process -FilePath $python -ArgumentList $trainingArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
if ($jobProcess.HasExited) { throw "Trainer exited early; inspect $stderr" }
$monitorArguments = @('-X', 'utf8', '-u', '-m', 'scripts.monitor_stage1', '--pid', [string]$jobProcess.Id,
    '--output', $telemetry, '--interval', '10')
$monitorProcess = Start-Process -FilePath $python -ArgumentList $monitorArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $monitorStdout -RedirectStandardError $monitorStderr -PassThru
$record = [ordered]@{stage='pretrain'; resume=$checkpoint; prior_updates=$ExpectedStep; prior_loss_tokens=$ExpectedTokens;
    pid=$jobProcess.Id; process_create_time_utc=$jobProcess.StartTime.ToUniversalTime().ToString('o');
    monitor_pid=$monitorProcess.Id; started_utc=[DateTime]::UtcNow.ToString('o'); arguments=$trainingArguments;
    micro_batch=8; grad_accum=8; seed=42; stdout=$stdout; stderr=$stderr; telemetry=$telemetry;
    budget="100000000 total valid next-token targets including the paused $ExpectedTokens"; record=$launchRecord}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchRecord -Encoding utf8
[pscustomobject]$record | Select-Object stage,pid,monitor_pid,started_utc,stdout,stderr,telemetry,record,budget | ConvertTo-Json
