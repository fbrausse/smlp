#!/usr/bin/env bash
set -euo pipefail

if ! xset q &>/dev/null; then
    pkill Xvfb &>/dev/null || true
    export DISPLAY=:99
    Xvfb "$DISPLAY" -screen 0 1024x768x16 &>Xvfb.log &

    i=1
    while true; do
        sleep 1
        if xset q &>/dev/null; then
            break
        fi
        if (( i < 11 )); then
            (( i++ ))
        else
            echo -e "\nERROR: Can't open virtual display\n" >&2
            exit 1
        fi
    done
fi
