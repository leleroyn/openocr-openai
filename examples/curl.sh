#!/bin/bash
# curl 调用示例: bash examples/curl.sh <图片路径>
# 环境变量: OCR_BASE(默认 http://127.0.0.1:8089/v1)  OCR_MODEL(默认 scriptmoe)
set -euo pipefail

BASE="${OCR_BASE:-http://127.0.0.1:8089/v1}"
MODEL="${OCR_MODEL:-scriptmoe}"   # 旧 id opendoc-0.1b / unirec-0.1b 走别名照样能用
IMG="${1:?用法: bash examples/curl.sh <图片路径>}"

echo "== /health =="
curl -s "${BASE%/v1}/health"; echo
echo "== /v1/models =="
curl -s "$BASE/models"; echo

b64=$(base64 -w0 "$IMG" 2>/dev/null || base64 -i "$IMG" | tr -d '\n')
echo "== /v1/chat/completions ($MODEL) =="
curl -s "$BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$b64\"}}]}]}"
echo
