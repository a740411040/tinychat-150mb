param([Parameter(Mandatory)][ValidateSet('pretrain', 'old_sft', 'clean_sft')][string]$Stage)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$config = Join-Path $projectRoot 'configs\pilot.json'
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONPATH = $projectRoot

if ($Stage -eq 'pretrain') {
    $trainingStage = 'pretrain'
    $dataDirectory = Join-Path $projectRoot 'data\pretrain_clean_v2_wiki'
    $runDirectory = Join-Path $projectRoot 'runs\pretrain_clean_100m_v2'
    $reviewMarker = Join-Path $projectRoot 'reports\comparison_v2\predata_wiki_review_ready.txt'
    $extraArguments = @('--target-tokens', '100000000')
} else {
    $trainingStage = 'sft'
    $dataDirectory = Join-Path $projectRoot 'data\sft_verified_v2'
    $reviewMarker = Join-Path $projectRoot 'reports\comparison_v2\verified_sft_review_ready.txt'
    if ($Stage -eq 'old_sft') {
        $runDirectory = Join-Path $projectRoot 'runs\sft_oldbase_v2'
        $baseCheckpoint = Join-Path $projectRoot 'runs\pretrain_100m_v1\step_00003052.pt'
    } else {
        $runDirectory = Join-Path $projectRoot 'runs\sft_cleanbase_v2'
        $baseCheckpoint = Join-Path $projectRoot 'runs\pretrain_clean_100m_v2\step_00003052.pt'
    }
    if (-not (Test-Path -LiteralPath $baseCheckpoint -PathType Leaf)) { throw "Missing base checkpoint: $baseCheckpoint" }
    $extraArguments = @('--init-checkpoint', $baseCheckpoint)
}
if (-not (Test-Path -LiteralPath $reviewMarker -PathType Leaf)) { throw "Quality review has not released $Stage" }
$manifestPath = Join-Path $dataDirectory 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Missing frozen manifest: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($manifest.frozen -ne $true) { throw 'Corpus is not frozen' }
if ($trainingStage -eq 'sft' -and ($manifest.splits.train.documents -ne 20000 -or $manifest.counts.train -ne 20000)) {
    throw 'SFT budget requires exactly 20000 frozen training examples'
}
if (Test-Path -LiteralPath $runDirectory) { throw "Refusing an existing run directory: $runDirectory" }
$active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($projectRoot) -and $_.CommandLine.Contains('tinychat.train_stage1')
})
if ($active.Count) { throw 'An owned training process is already running' }
$logRoot = Join-Path $projectRoot 'logs'
$stdout = Join-Path $logRoot "comparison_$Stage.stdout.log"
$stderr = Join-Path $logRoot "comparison_$Stage.stderr.log"
$telemetry = Join-Path $logRoot "comparison_$Stage.gpu.csv"
$monitorStdout = Join-Path $logRoot "comparison_$Stage.monitor.log"
$monitorStderr = Join-Path $logRoot "comparison_$Stage.monitor.err.log"
foreach ($target in @($stdout, $stderr, $telemetry, $monitorStdout, $monitorStderr)) {
    if (Test-Path -LiteralPath $target) { throw "Refusing existing job log: $target" }
}
New-Item -ItemType Directory -Path $runDirectory | Out-Null
$trainingArguments = @('-X', 'utf8', '-u', '-m', 'tinychat.train_stage1', '--stage', $trainingStage,
    '--data-dir', $dataDirectory, '--run-dir', $runDirectory, '--config', $config,
    '--micro-batch', '8', '--grad-accum', '8', '--checkpoint-every', '250', '--eval-every', '250',
    '--device', 'cuda', '--seed', '42') + $extraArguments
$jobProcess = Start-Process -FilePath $python -ArgumentList $trainingArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
if ($jobProcess.HasExited) { throw "Trainer exited early; inspect $stderr" }
$monitorArguments = @('-X', 'utf8', '-u', '-m', 'scripts.monitor_stage1', '--pid', [string]$jobProcess.Id,
    '--output', $telemetry, '--interval', '10')
$monitorProcess = Start-Process -FilePath $python -ArgumentList $monitorArguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $monitorStdout `
    -RedirectStandardError $monitorStderr -PassThru
$record = [ordered]@{ stage = $Stage; pid = $jobProcess.Id; process_create_time_utc = $jobProcess.StartTime.ToUniversalTime().ToString('o');
    monitor_pid = $monitorProcess.Id; started_utc = [DateTime]::UtcNow.ToString('o'); arguments = $trainingArguments;
    micro_batch = 8; grad_accum = 8; seed = 42; stdout = $stdout; stderr = $stderr; telemetry = $telemetry;
    budget = $(if ($Stage -eq 'pretrain') { '100000000 valid next-token targets' } else { '20000 examples, one epoch' }) }
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runDirectory 'launcher.json') -Encoding utf8
([pscustomobject]$record) | Select-Object stage, pid, monitor_pid, started_utc, budget | ConvertTo-Json
