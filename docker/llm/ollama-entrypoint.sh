#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5
IFS=',' read -ra MODEL_ARRAY <<< "$OLLAMA_MODEL_DL"

# Loop over each model name
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
  echo "Found model: $MODEL_NAME"
  echo "🔴 Retrieve $MODEL_NAME model..."
  ollama pull $MODEL_NAME
done
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
