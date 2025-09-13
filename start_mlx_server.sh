#!/bin/zsh

# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/start_mlx_server.sh

# Path to the directory containing the standard mlx_lm package (for dependencies)
MLX_LM_DIR=/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-lm
# Path to the custom server.py
CUSTOM_SERVER=/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/server/base/server.py

# Ensure the mlx_lm package is accessible
export PYTHONPATH=$MLX_LM_DIR:$PYTHONPATH

# Default models and settings
DEFAULT_MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_DRAFT_MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
DEFAULT_NUM_DRAFT_TOKENS=5
DEFAULT_MAX_TOKENS=1024
DEFAULT_TEMP=0.0
DEFAULT_TOP_K=1
DEFAULT_LOG_LEVEL="DEBUG"

# Parse command-line arguments
MODEL="$DEFAULT_MODEL"
DRAFT_MODEL="$DEFAULT_DRAFT_MODEL"
NUM_DRAFT_TOKENS="$DEFAULT_NUM_DRAFT_TOKENS"
MAX_TOKENS="$DEFAULT_MAX_TOKENS"
TEMP="$DEFAULT_TEMP"
TOP_K="$DEFAULT_TOP_K"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        --num-draft-tokens)
            NUM_DRAFT_TOKENS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log the configuration
echo "Using main model: $MODEL"
echo "Using draft model: $DRAFT_MODEL"
echo "Speculative decoding tokens: $NUM_DRAFT_TOKENS"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMP"
echo "Top-k: $TOP_K"
echo "Log level: $LOG_LEVEL"

# Verify PYTHONPATH
# echo "PYTHONPATH: $PYTHONPATH"

# Run the server with optimized settings
echo "Executing command: python $CUSTOM_SERVER --host 0.0.0.0 --port 8080 --use-default-chat-template --model $MODEL --draft-model $DRAFT_MODEL --num-draft-tokens $NUM_DRAFT_TOKENS --max-tokens $MAX_TOKENS --temp $TEMP --top-k $TOP_K --log-level $LOG_LEVEL"
python "$CUSTOM_SERVER" --host 0.0.0.0 --port 8080 --use-default-chat-template --model "$MODEL" --draft-model "$DRAFT_MODEL" --num-draft-tokens "$NUM_DRAFT_TOKENS" --max-tokens "$MAX_TOKENS" --temp "$TEMP" --top-k "$TOP_K" --log-level "$LOG_LEVEL"