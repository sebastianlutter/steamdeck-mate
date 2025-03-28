#!/bin/bash
# Get a shell in the running ollama docker
CONTAINER_ID=$(docker ps --filter "name=ollama-llm" --format "{{.ID}}")
docker exec -it $CONTAINER_ID bash
