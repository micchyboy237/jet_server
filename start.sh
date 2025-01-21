#!/bin/zsh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Start the uvicorn server
# uvicorn app:app --host 0.0.0.0 --port 8002 --reload --reload-dir /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules
python app.py
