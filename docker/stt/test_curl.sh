#!/bin/bash
cd "$(dirname $0)"

if [ "$1" == "" ]; then
  file="audio.wav"
else
  file="$1"
fi
curl -v http://localhost:8000/v1/audio/transcriptions -F "file=@${file}" -F "stream=true"
