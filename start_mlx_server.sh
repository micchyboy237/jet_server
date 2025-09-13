# mlx_lm.server --host 0.0.0.0 --port 8080 --use-default-chat-template --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Path to the directory containing the standard mlx_lm package (for dependencies)
MLX_LM_DIR=/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/mlx-lm
# Path to the custom server.py
CUSTOM_SERVER=/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/server/base/server.py

# Ensure the mlx_lm package is accessible
export PYTHONPATH=$MLX_LM_DIR:$PYTHONPATH

# Default model
DEFAULT_MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"

# Parse command-line arguments
MODEL="$DEFAULT_MODEL"
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log the model being used
echo "Using model: $MODEL"

# Verify PYTHONPATH
# echo "PYTHONPATH: $PYTHONPATH"

# Run the server directly with the specified or default model
echo "Executing command: python $CUSTOM_SERVER --host 0.0.0.0 --port 8080 --use-default-chat-template --model $MODEL"
python "$CUSTOM_SERVER" --host 0.0.0.0 --port 8080 --use-default-chat-template --model "$MODEL"
