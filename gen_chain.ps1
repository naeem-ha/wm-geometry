# Detached generation chain. Resumable: run.py skips clips that already have meta.json.
# Log: work/gen_chain.log.  Usage: gen_chain.ps1 [-Models wan22_5b] [-Limit 25]
param([string[]]$Models = @("cogvideox_5b", "wan22_5b"), [int]$Limit = 40)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
foreach ($m in $Models) {
    "$(Get-Date -Format s) start $m (limit $Limit)" | Out-File -Append -Encoding utf8 work\gen_chain.log
    & $py run.py generate --model $m --limit $Limit 2>&1 | Where-Object { $_ -match '^droid_|Traceback|Error' } |
        Out-File -Append -Encoding utf8 work\gen_chain.log
    "$(Get-Date -Format s) done $m (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 work\gen_chain.log
}
