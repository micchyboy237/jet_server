#!/bin/zsh

# Set MLX_SERVER_DIR
MLX_SERVER_DIR="/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-lm/mlx_lm"

# Function to check if the server is running
check_server_running() {
  if curl -s http://127.0.0.1:8003/health | grep -q '"status": "ok"'; then
    echo "[INFO] MLX server is already running on port 8003."
    return 0
  else
    echo "[INFO] No MLX server detected on port 8003."
    return 1
  fi
}

# Function to wait for the server to start
wait_for_server() {
  local max_attempts=30
  local attempt=1
  echo "[INFO] Waiting for MLX server to start on port 8003..."
  while ! curl -s http://127.0.0.1:8003/health | grep -q '"status": "ok"'; do
    if [ $attempt -ge $max_attempts ]; then
      echo "[ERROR] MLX server failed to start after $max_attempts attempts."
      exit 1
    fi
    echo "[INFO] Attempt $attempt: Server not ready, retrying in 1 second..."
    sleep 1
    ((attempt++))
  done
  echo "[INFO] MLX server is ready."
}

# Verify MLX_SERVER_DIR exists
if [ ! -d "$MLX_SERVER_DIR" ]; then
  echo "[ERROR] Directory $MLX_SERVER_DIR does not exist."
  exit 1
fi

# Start the MLX server only if it's not already running
if ! check_server_running; then
  echo "[INFO] Starting MLX server in a new Terminal window..."
  osascript -e 'tell application "Terminal" to do script "
    cd \"'"$MLX_SERVER_DIR"'\"
    python -m mlx_lm server --model \"mlx-community/Llama-3.2-3B-Instruct-4bit\" --port 8003
    echo \"[INFO] MLX server stopped.\"
    exit"'
  # Wait for the server to be ready
  wait_for_server
else
  echo "[INFO] Using existing MLX server instance."
fi

# 1. Text Completion Request
curl -X POST http://127.0.0.1:8003/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'

# 2. Chat Completion Request (Non-Streaming)
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 20,
    "temperature": 0.5,
    "logprobs": 5
  }'

# 3. Chat Completion with Streaming
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a short story about a robot."}
    ],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 100,
    "temperature": 0.8
  }'

# 4. List Available Models
curl -X GET http://127.0.0.1:8003/v1/models

# 5. Health Check
curl -X GET http://127.0.0.1:8003/health

# 6. Chat Completion with Stop Words and Logit Bias
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Describe a sunny day."}
    ],
    "max_tokens": 50,
    "stop": ["rain", "storm"],
    "logit_bias": {"12345": 10.0, "67890": -10.0},
    "temperature": 0.6
  }'

# 7. Using a Custom Model and Adapter
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "adapters": "./fine_tuned_adapter",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'