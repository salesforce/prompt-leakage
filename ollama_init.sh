#!/bin/bash
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
echo "sleep for 100"
sleep 100
ollama pull llama2:13b-chat
ollama pull mistral:v0.2
ollama pull mixtral:8x7b
