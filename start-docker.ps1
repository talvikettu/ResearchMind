param([switch]$Gpu, [switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$containerIds = @(docker ps -a -q)
if ($LASTEXITCODE -ne 0) { throw 'Docker is unavailable. Start Docker Desktop.' }
$reusable = @()
foreach ($containerId in $containerIds) {
    $item = (docker inspect $containerId | ConvertFrom-Json)[0]
    if ($item.Config.Labels.'com.docker.compose.project' -eq 'researchmind' -and
        $item.Config.Labels.'com.docker.compose.service' -in @('app','qdrant','ollama')) {
        $reusable += $item.Config.Labels.'com.docker.compose.service'
    }
}
$additional = 3 - @($reusable | Select-Object -Unique).Count
if ($containerIds.Count + $additional -gt 3) {
    throw "Container limit is 3 in total, including stopped containers. Existing: $($containerIds.Count); required additional: $additional. Reuse/remove your unneeded containers first, or use .\start.ps1."
}
$composeArguments = @('compose','-p','researchmind','-f','compose.yaml')
if ($Gpu) { $composeArguments += @('-f','compose.gpu.yaml') }
$composeArguments += @('up','-d','--no-recreate')
if ($Build) { $composeArguments += '--build' } else { $composeArguments += '--no-build' }
& docker @composeArguments
if ($LASTEXITCODE -ne 0) { throw 'Docker startup failed. For a missing application image use -Build.' }
Write-Host 'UI: http://127.0.0.1:8501 (or RESEARCH_UI_PORT). Logs: docker compose -p researchmind logs -f app'
