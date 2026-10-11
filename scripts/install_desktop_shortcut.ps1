# Installe un raccourci Â« MIKATA Â» sur le Bureau : double-clic = dÃ©marrage rapide.
# MÃªme lancement que le dÃ©marrage automatique : pythonw (pas de console), MIKATA
# tourne en arriÃ¨re-plan avec sa mascotte et sa barre de menus.
$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $project '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $pythonw)) {
    throw "pythonw introuvable : $pythonw (lance `"uv sync`" d'abord)."
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'MIKATA.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '-m mikata'
$shortcut.WorkingDirectory = $project
$shortcut.Description = 'MIKATA - assistant vocal'
$shortcut.Save()

Write-Host "Raccourci cree : $link"
