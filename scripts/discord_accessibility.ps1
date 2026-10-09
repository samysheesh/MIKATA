# Discord lisible par MIKATA (phase 12a) : ajoute --force-renderer-accessibility aux
# raccourcis de Discord et à son démarrage automatique. Electron/Chromium n'expose son
# arbre d'accessibilité Windows (salon vocal, membres...) que lancé avec ce drapeau ; le
# stub Discord.exe (Squirrel) le transmet au vrai processus, y compris après les mises à
# jour.
#
#   powershell -ExecutionPolicy Bypass -File scripts\discord_accessibility.ps1           # installe
#   powershell -ExecutionPolicy Bypass -File scripts\discord_accessibility.ps1 -Restore  # retire
#   powershell -ExecutionPolicy Bypass -File scripts\discord_accessibility.ps1 -Status   # état
#
# Après l'installation : quitte Discord complètement (icône barre des tâches > Quitter
# Discord) puis relance-le depuis le raccourci. L'icône de la barre des tâches peut
# demander un redémarrage de l'Explorateur pour être mise à jour.

param(
    [switch]$Restore,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$flag = "--force-renderer-accessibility"
$stub = Join-Path $env:LOCALAPPDATA "Discord\Discord.exe"

if (-not (Test-Path $stub)) {
    throw "Discord introuvable : $stub"
}

$appData = $env:APPDATA
$links = @(
    "$appData\Microsoft\Windows\Start Menu\Programs\Discord.lnk",
    "$appData\Microsoft\Windows\Start Menu\Programs\Discord Inc\Discord.lnk",
    "$appData\Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar\Discord.lnk"
) | Where-Object { Test-Path $_ }

if (-not $links) {
    throw "aucun raccourci Discord trouvé (menu Démarrer / barre des tâches)"
}

$shell = New-Object -ComObject WScript.Shell

if ($Status) {
    foreach ($lnk in $links) {
        $shortcut = $shell.CreateShortcut($lnk)
        $ok = ($shortcut.TargetPath -eq $stub) -and ($shortcut.Arguments -like "*$flag*")
        if ($ok) {
            Write-Output ("{0} : drapeau présent" -f (Split-Path $lnk -Leaf))
        } else {
            Write-Output ("{0} : drapeau ABSENT (cible={1}, arguments={2})" -f `
                (Split-Path $lnk -Leaf), $shortcut.TargetPath, $shortcut.Arguments)
        }
    }
} else {
    foreach ($lnk in $links) {
        $shortcut = $shell.CreateShortcut($lnk)
        $target = $stub
        $arguments = if ($Restore) { "" } else { $flag }
        if ($shortcut.TargetPath -eq $target -and $shortcut.Arguments -eq $arguments) {
            continue  # déjà dans l'état voulu
        }
        $shortcut.TargetPath = $target
        $shortcut.Arguments = $arguments
        $shortcut.Save()
        Write-Output ("{0} : {1}" -f (Split-Path $lnk -Leaf), `
            $(if ($Restore) { "drapeau retiré" } else { "drapeau ajouté" }))
    }
}

# Démarrage automatique de Windows (registre) : Update.exe transmet ses arguments au vrai
# processus via --process-start-args ; le drapeau y est inséré (retiré par -Restore), sinon
# Discord reviendrait en lecture illisible à chaque ouverture de session.
$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$entry = Get-ItemProperty -Path $runKey -Name Discord -ErrorAction SilentlyContinue
if ($null -ne $entry) {
    $value = [string]$entry.Discord
    if ($Status) {
        $ok = $value -like "*$flag*"
        Write-Output ("démarrage automatique : {0}" -f `
            $(if ($ok) { "drapeau présent" } else { "drapeau ABSENT" }))
    } elseif ($value -match '--process-start-args\s+"([^"]*)"') {
        $inner = $Matches[1]
        if ($Restore) {
            $newInner = (($inner -split " ") | Where-Object { $_ -and $_ -ne $flag }) -join " "
        } elseif ($inner -notlike "*$flag*") {
            $newInner = "$inner $flag"
        } else {
            $newInner = $inner
        }
        if ($newInner -cne $inner) {
            $newValue = $value.Replace($inner, $newInner)
            Set-ItemProperty -Path $runKey -Name Discord -Value $newValue
            Write-Output ("démarrage automatique : {0}" -f `
                $(if ($Restore) { "drapeau retiré" } else { "drapeau ajouté" }))
        }
    }
}

if (-not $Restore -and -not $Status) {
    Write-Output "Fait. Quitte Discord complètement puis relance-le depuis le raccourci."
}
