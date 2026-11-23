$ErrorActionPreference = 'Stop'
$source = Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\Resume-ComparisonPretraining.ps1'
$ast = [Management.Automation.Language.Parser]::ParseFile($source, [ref]$null, [ref]$null)
$guardAst = $ast.Find({param($node) $node -is [Management.Automation.Language.IfStatementAst]}, $true)
$guard = [scriptblock]::Create($guardAst.Extent.Text)
$ExpectedStep = 1004
$ExpectedTokens = 32899072
$checkpoint = 'fixture.pt'
$cases = @(
    @{name='safe'; change=@{}; remove=@(); accept=$true},
    @{name='missing_safe_flag'; change=@{}; remove=@('optimizer_state_unsafe'); accept=$false},
    @{name='null_safe_flag'; change=@{optimizer_state_unsafe=$null}; remove=@(); accept=$false},
    @{name='string_safe_flag'; change=@{optimizer_state_unsafe='false'}; remove=@(); accept=$false},
    @{name='unsafe_flag'; change=@{optimizer_state_unsafe=$true}; remove=@(); accept=$false},
    @{name='recovery_error'; change=@{recovery_error='failed save'}; remove=@(); accept=$false},
    @{name='missing_recovery_error'; change=@{}; remove=@('recovery_error'); accept=$false},
    @{name='wrong_step'; change=@{checkpoint_step=255}; remove=@(); accept=$false},
    @{name='wrong_tokens'; change=@{cumulative_valid_loss_tokens=8355840}; remove=@(); accept=$false}
)
foreach ($case in $cases) {
    $fields = @{stop_reason='interrupted'; optimizer_state_unsafe=$false; recovery_error=$null;
        checkpoint_step=1004; cumulative_valid_loss_tokens=32899072; planned_loss_tokens=100000000; checkpoint=$checkpoint}
    foreach ($key in $case.change.Keys) { $fields[$key] = $case.change[$key] }
    foreach ($key in $case.remove) { $fields.Remove($key) }
    $metrics = [pscustomobject]$fields
    $accepted = $true
    try { & $guard } catch { $accepted = $false }
    if ($accepted -ne $case.accept) { throw "Unexpected guard result: $($case.name)" }
}
[pscustomobject]@{cases=$cases.Count; passed=$cases.Count; scope='actual metadata guard only; no trainer or GPU started'} | ConvertTo-Json
