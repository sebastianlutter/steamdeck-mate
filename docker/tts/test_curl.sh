#!/bin/bash

WORDS="Tri tra die Hex ist tot."

for model in thorsten-low ramona-low thorsten-medium thorsten-medium-emo thorsten-high; do
 curl http://localhost:8001/v1/audio/speech -H "Content-Type: application/json" -d "{
    \"model\": \"tts-1\",
    \"input\": \"${WORDS}\",
    \"voice\": \"${model}\",
    \"response_format\": \"wav\",
    \"speed\": 1.0
  }" > speech_${model}.mp3
done