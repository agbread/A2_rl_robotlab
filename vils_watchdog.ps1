# VILS watchdog: stop GPU training while VILS is running, resume it when VILS exits.
# Stop this watchdog by creating a file named stop_watchdog.txt in the repo root.
$root = 'C:\Users\user\go2_rl_robotlab'
$log  = Join-Path $root 'logs\watchdog.log'
$stop = Join-Path $root 'stop_watchdog.txt'

function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content $log }

Log "watchdog started (pid $PID)"
while ($true) {
    if (Test-Path $stop) { Log "stop_watchdog.txt found; exiting"; break }

    $vils  = Get-Process -Name VILS  -ErrorAction SilentlyContinue
    $train = Get-Process -Name python -ErrorAction SilentlyContinue

    if ($vils -and $train) {
        Log "VILS detected (pid $($vils.Id)) -> stopping training (pid $($train.Id)); it will resume from the latest checkpoint after VILS exits"
        Stop-Process -Name python -Force -ErrorAction SilentlyContinue
    }
    elseif (-not $vils -and -not $train) {
        Log "VILS gone and no training running -> relaunching training"
        Start-Process -FilePath (Join-Path $root 'launch_go2_train.bat') -WindowStyle Hidden
        Start-Sleep -Seconds 300   # give Isaac Sim time to start before checking again
    }

    Start-Sleep -Seconds 15
}
