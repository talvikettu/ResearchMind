param([switch]$RefreshLibrary)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
try {
    $running = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3
    if ($running.embedding_model) {
        Write-Host 'ResearchMind API is already running. Use .\stop.ps1 before changing configuration.'
        Write-Host 'UI: http://127.0.0.1:8501'
        return
    }
} catch { }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
uv sync --frozen --extra dev --extra models
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$config = (& $python -c 'import json; from researchmind.config import Settings; s=Settings(); print(json.dumps(dict(backend=s.answer_backend,url=s.ollama_url)))') | ConvertFrom-Json
$logs = Join-Path $PSScriptRoot 'data\logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$started = @()
function Start-Worker($executable, $arguments, $label) {
    $process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs "$label.log") -RedirectStandardError (Join-Path $logs "$label-error.log")
    return $process.Id
}
function Wait-Endpoint($url, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod -Uri $url -TimeoutSec 3 | Out-Null; return } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "Service did not become ready: $url. See data/logs."
}
if ($config.backend -eq 'ollama') {
    $ready = $false
    try { Invoke-RestMethod -Uri "$($config.url)/api/tags" -TimeoutSec 3 | Out-Null; $ready = $true } catch { }
    if (-not $ready) {
        $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollamaCommand) { throw 'Install Ollama from https://ollama.com/download/windows or use Docker.' }
        $uri = [Uri]$config.url
        if ($uri.Host -notin @('127.0.0.1','localhost')) { throw 'Start the configured remote Ollama service first.' }
        $env:OLLAMA_HOST = "$($uri.Host):$($uri.Port)"
        $env:OLLAMA_MODELS = Join-Path $PSScriptRoot 'data\models\ollama'
        $started += Start-Worker $ollamaCommand.Source 'serve' 'ollama'
        Wait-Endpoint "$($config.url)/api/tags" 60
    }
}
$bootstrapArgs = @('-m','researchmind.bootstrap')
if ($RefreshLibrary) { $bootstrapArgs += '--download' }
& $python @bootstrapArgs
if ($LASTEXITCODE -ne 0) { throw 'Library preparation failed. See data/index-status.json and data/papers/import-status.json.' }
$started += Start-Worker $python '-m uvicorn researchmind.api:app --host 127.0.0.1 --port 8000' 'api'
@{pids=$started; root=$PSScriptRoot} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'data\runtime.json') -Encoding UTF8
Wait-Endpoint 'http://127.0.0.1:8000/health' 180
$started += Start-Worker $python '-m streamlit run ui.py --server.address 127.0.0.1 --server.port 8501 --server.headless true' 'ui'
@{pids=$started; root=$PSScriptRoot} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'data\runtime.json') -Encoding UTF8
Wait-Endpoint 'http://127.0.0.1:8501/_stcore/health' 60
Write-Host 'ResearchMind: http://127.0.0.1:8501'
Write-Host 'Stop: .\stop.ps1. Logs: data/logs.'
