#!/bin/zsh

export PYTHONPATH="$PYTHONPATH:$PWD"

JET_SERVER_DIR="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server"
MLX_SERVER_DIR="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/server"

# Function to clean up all processes
cleanup() {
  echo "[INFO] Shutting down all processes..."
  # Kill the MLX server if it exists
  if [[ -n "$MLX_PID" ]]; then
    kill "$MLX_PID" 2>/dev/null
  fi
  # Kill the main Python server (will be killed automatically, but ensure cleanup)
  exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT

# Check if port 9000 is in use
if ! lsof -i :9000 >/dev/null; then
  echo "[INFO] Port 9000 is free. Starting MLX server in a new Terminal window..."
  osascript -e 'tell application "Terminal" to do script "
    cd \"'"$MLX_SERVER_DIR"'\"
    ./start.sh
    echo \"[INFO] MLX server stopped.\"
    exit"'
else
  echo "[INFO] Port 9000 is already in use. Skipping MLX server startup."
fi

echo "[INFO] Starting Python server in current terminal..."
python app.py &

# Store the PID of the main Python server
MAIN_PID=$!

# Wait for the main server to exit
wait $MAIN_PID

# Call cleanup when the main server exits
cleanup