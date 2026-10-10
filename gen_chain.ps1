# Detached generation chain. Resumable: run.py skips clips that already have meta.json.
# Log: work/gen_chain.log.  Usage: gen_chain.ps1 -Jobs ctrlworld:60,wan22_5b:25
param([string[]]$Jobs = @("cogvideox_5b:40", "wan22_5b:40"))
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
# -File passes "a:1,b:2" as one string, so split on commas here
foreach ($job in ($Jobs -join ",").Split(",")) {
    $m, $limit = $job.Split(":")
    "$(Get-Date -Format s) start $m (limit $limit)" | Out-File -Append -Encoding utf8 work\gen_chain.log
    & $py run.py generate --model $m --limit $limit 2>&1 | Where-Object { $_ -match '^droid_|Traceback|Error' } |
        Out-File -Append -Encoding utf8 work\gen_chain.log
    "$(Get-Date -Format s) done $m (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 work\gen_chain.log
}
