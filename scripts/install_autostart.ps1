<#
.SYNOPSIS
    Démarrage automatique de MIKATA à l'ouverture de session — phase 7 (§10).

.DESCRIPTION
    Crée (ou remplace) la tâche planifiée « MIKATA » :
    - déclencheur : ouverture de TA session, 30 s après (micro et casque prêts) ;
    - action : .venv\Scripts\pythonw.exe -m mikata --supervise, dans le dossier du projet
      (pas de console ; MIKATA est relancé s'il plante, pas si tu fais « Quitter ») ;
    - ton compte, droits normaux (jamais administrateur), sans limite de durée, aussi sur
      batterie, une seule instance à la fois ; « redémarrer en cas d'échec » activé.
    Pas un service Windows : un service tourne dans la session 0, sans accès à ton bureau ni
    à ton audio.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -StartNow
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Status
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall
#>
param(
    [switch]$Uninstall,
    [switch]$Status,
    [switch]$StartNow,
    [string]$TaskName = "MIKATA",
    [int]$DelaySeconds = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$Arguments = "-m mikata --supervise"

if ($Status) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "Tâche « $TaskName » absente : MIKATA ne démarre pas tout seul."
        exit 1
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "Tâche « $TaskName » : $($task.State)"
    Write-Host "  Action : $($task.Actions[0].Execute) $($task.Actions[0].Arguments)"
    Write-Host "  Dernier lancement : $($info.LastRunTime) (résultat $($info.LastTaskResult))"
    Write-Host "  Prochain déclenchement : ouverture de session (+ $($task.Triggers[0].Delay))"
    exit 0
}

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Tâche « $TaskName » retirée : MIKATA ne démarrera plus tout seul."
    } else {
        Write-Host "Tâche « $TaskName » absente : rien à faire."
    }
    exit 0
}

if (-not (Test-Path $Pythonw)) {
    Write-Error "Introuvable : $Pythonw. Lance d'abord « uv sync » dans $ProjectRoot."
}

# Configuration validée avant d'installer : sinon la tâche échouerait en silence à chaque
# ouverture de session.
& $Python -m mikata --check-config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Configuration invalide : corrige config.yaml (message ci-dessus), puis relance."
}

$User = "$env:USERDOMAIN\$env:USERNAME"
$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument $Arguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$Trigger.Delay = "PT$($DelaySeconds)S"
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
        -Principal $Principal -Settings $Settings -Force `
        -Description "Assistant vocal MIKATA : démarre à l'ouverture de session." | Out-Null
} catch {
    Write-Error ("Création de la tâche refusée : $($_.Exception.Message)`n" +
        "Relance ce script une fois dans un PowerShell « Exécuter en tant qu'administrateur » " +
        "(la tâche tournera quand même avec tes droits normaux).")
}

Write-Host "Tâche « $TaskName » installée : MIKATA démarrera $DelaySeconds s après l'ouverture de ta session."
Write-Host "  Action : $Pythonw $Arguments"
Write-Host "  Dossier : $ProjectRoot"
Write-Host "  Journal des relances : data\logs\supervise.log"
if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Démarrée maintenant (l'icône apparaît dans la zone de notification)."
}
