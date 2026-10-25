$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$processes = @(Get-CimInstance Win32_Process)
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$targets = @($processes | Where-Object { $_.ExecutablePath -eq $venvPython -and $_.CommandLine -match 'uvicorn researchmind.api|streamlit run ui.py' } | ForEach-Object { $_.ProcessId })
# Descendants may use the shared base Python executable; only include children of our checkout's launcher.
for ($depth=0; $depth -lt 6; $depth++) {
    $targets += @($processes | Where-Object { $_.ParentProcessId -in $targets } | ForEach-Object { $_.ProcessId })
    $targets = @($targets | Select-Object -Unique)
}
foreach ($identifier in ($targets | Sort-Object -Descending)) { Stop-Process -Id $identifier -ErrorAction SilentlyContinue }
Write-Host 'ResearchMind API/UI stopped. The local model server stays available for the next launch.'
