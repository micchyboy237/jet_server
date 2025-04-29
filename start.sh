#!/bin/zsh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Function to clean up all processes
cleanup() {
  echo "[INFO] Shutting down all processes..."
  # Kill the main Python server (ensure cleanup)
  exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT

echo "[INFO] Starting Python server in current terminal..."
python app.py &

# Store the PID of the main Python server
MAIN_PID=$!

# Wait for the main server to exit
wait $MAIN_PID

# Call cleanup when the main server exits
cleanup
