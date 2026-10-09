# Installer le helper USN de MIKATA (phase 12d) : une tâche planifiée élevée qui lit la
# MFT et le journal de modifications des volumes NTFS, et dépose build.jsonl + events.log
# dans data/index/. MIKATA (utilisateur standard) ne lit que ces fichiers.
#
# Un seul UAC, à l'installation : exécuter ce script en administrateur.
#   PowerShell (admin) :  .\scripts\install_usn_helper.ps1
#   Désinstaller :        .\scripts\install_usn_helper.ps1 -Remove
param(
    [string]$Project = (Split-Path -Parent $PSScriptRoot),
    [switch]$Remove
)
$ErrorActionPreference = "Stop"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Élévation requise (un seul UAC)..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 0
}

$task = "MIKATA Indexer USN"
if ($Remove) {
    schtasks /Delete /TN $task /F 2>$null
    Write-Host "Tâche « $task » supprimée."
    exit 0
}

$data = Join-Path $Project "data"
$share = Join-Path $data "index"
$python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python du projet introuvable : $python (uv venv attendu)" }

New-Item -ItemType Directory -Force -Path $share | Out-Null
# L'utilisateur courant doit lire/écrire ce que le helper (admin) y dépose.
icacls $share /grant "$($env:USERNAME):(OI)(CI)M" /T /Q

$xmlPath = Join-Path $env:TEMP "mikata-usn-helper.xml"
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Index local des fichiers de MIKATA (lecture MFT + journal USN).</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger><UserId>$($env:USERNAME)</UserId><Enabled>true</Enabled></LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$($env:USERDOMAIN)\$($env:USERNAME)</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <StartWhenAvailable>true</StartWhenAvailable>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"$python"</Command>
      <Arguments>-m mikata.files.helper --data "$data"</Arguments>
      <WorkingDirectory>"$Project"</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
$xml | Out-File -FilePath $xmlPath -Encoding unicode
schtasks /Create /TN $task /XML $xmlPath /F
Remove-Item $xmlPath -Force
schtasks /Run /TN $task | Out-Null
Write-Host "Helper installé et lancé : $share" -ForegroundColor Green
Write-Host "Vérifie data/index/helper.log et data/index/build.jsonl dans quelques secondes."
