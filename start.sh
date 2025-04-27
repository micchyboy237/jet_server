#!/bin/zsh

export PYTHONPATH="$PYTHONPATH:$PWD"

JET_SERVER_DIR="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server"
MLX_SERVER_DIR="/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-lm/mlx_lm"


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

# Function to clean up all processes
cleanup() {
  echo "[INFO] Shutting down all processes..."
  # Kill the MLX server if it exists
  if [[ -n "$MLX_PID" ]]; then
    kill "$MLX_PID" 2>/dev/null
  fi
  # Kill the LocalTunnel keep-alive ping if it exists
  if [[ -n "$PING_PID" ]]; then
    kill "$PING_PID" 2>/dev/null
  fi
  # Kill the main Python server (will be killed automatically, but ensure cleanup)
  exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT

if [ "$START_TUNNEL" = true ]; then
  echo "[INFO] Starting persistent LocalTunnel on subdomain 'jetdev' in a new Terminal window..."
  osascript -e 'tell application "Terminal" to do script "
    cd \"'"$JET_SERVER_DIR"'\"
    while true; do
      echo \"[INFO] Launching LocalTunnel...\"
      lt --port 8002 --subdomain jetdev
      echo \"[WARN] LocalTunnel disconnected. Restarting...\"
    done"'
  
  echo "[INFO] Starting keep-alive ping to http://jetdev.loca.lt every 30 seconds..."
  (while true; do curl --silent http://jetdev.loca.lt > /dev/null; sleep 30; done) &
  PING_PID=$!
else
  echo "[INFO] LocalTunnel not started. Use -s to enable."
fi

echo "[INFO] Starting MLX server in a new Terminal window..."
osascript -e 'tell application "Terminal" to do script "
  cd \"'"$MLX_SERVER_DIR"'\"
  python -m mlx_lm.server --model \"mlx-community/Llama-3.2-3B-Instruct-4bit\" -port 8003
  echo \"[INFO] MLX server stopped.\"
  exit"'

echo "[INFO] Starting Python server in current terminal..."
python app.py &

# Store the PID of the main Python server
MAIN_PID=$!

# Wait for the main server to exit
wait $MAIN_PID

# Call cleanup when the main server exits
cleanup