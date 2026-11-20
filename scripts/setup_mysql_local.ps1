$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$setupPath = Join-Path $PSScriptRoot 'setup_mysql_local.py'
$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList @('-NoProfile','-NoExit','-ExecutionPolicy','Bypass','-File',('"' + $PSCommandPath + '"'))
    exit
}
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = 'Local MySQL setup - secure password input'
& $pythonPath -X utf8 $setupPath
Write-Host 'Setup finished. You may close this window. Do not paste passwords into chat.'
