#!/bin/zsh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Parse arguments
START_TUNNEL=false

while getopts ":s" opt; do
  case ${opt} in
    s )
      START_TUNNEL=true
      ;;
    \? )
      echo "Usage: ./start.sh [-s]"
      exit 1
      ;;
  esac
done

if [ "$START_TUNNEL" = true ]; then
  echo "[INFO] Starting persistent LocalTunnel on subdomain 'jetdev' in a new Terminal window..."
  osascript -e 'tell application "Terminal" to do script "
    cd \"'"$PWD"'\"
    while true; do
      echo \"[INFO] Launching LocalTunnel...\"
      lt --port 8002 --subdomain jetdev
      echo \"[WARN] LocalTunnel disconnected. Restarting in 5 seconds...\"
      sleep 5
    done"'
else
  echo "[INFO] LocalTunnel not started. Use -s to enable."
fi

echo "[INFO] Starting Python server in current terminal..."
python app.py
