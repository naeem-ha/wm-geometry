# Detached scoring after generation: waits for generation and any other scoring to exit, then scores the
# remaining clips on both tracks and runs VGGT (GPU) on every condition. Log: work/score_chain.log
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$log = "work\score_chain.log"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run\.py (generate|score)' }) { Start-Sleep 30 }
foreach ($step in @(@("--no-vggt"), @("--fps5"), @())) {
    "$(Get-Date -Format s) start score $($step -join ' ')" | Out-File -Append -Encoding utf8 $log
    & $py run.py score @step 2>&1 | Where-Object { $_ -match 'Traceback|Error' } | Out-File -Append -Encoding utf8 $log
    "$(Get-Date -Format s) done score $($step -join ' ') (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
