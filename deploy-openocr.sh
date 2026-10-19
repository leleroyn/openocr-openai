#!/bin/bash
# ============================================================================
# 发布/重建 openocr-openai（OpenDoc-0.1B + UniRec-0.1B 的 OpenAI 兼容接口）
#
# 用法:
#   bash deploy-openocr.sh                        # 重建容器（CPU 模式，默认）+ 健康检查
#   bash deploy-openocr.sh --gpu                  # GPU 模式（需空闲显存 >=4.5GB）
#   bash deploy-openocr.sh --image 1.8            # 指定镜像 tag
#   bash deploy-openocr.sh --max-pixel-side 832   # 前置缩放上限（默认 1024px，越小越省显存）
#   bash deploy-openocr.sh --port 8090            # 换端口（试验用）
#   bash deploy-openocr.sh --model-dir /data/ocr/openocr-openai/models
#   bash deploy-openocr.sh --cpuset 0-7            # 绑到 P-core（共享机推荐）
#   bash deploy-openocr.sh --dry-run              # 只打印 docker run 命令，不执行
#
# 说明:
#   1) 镜像在本地构建，docker save + 上传，生产服务器只 load，禁止在服务器构建。
#   2) 模型缓存在 --model-dir（挂载进容器 /root/.cache/openocr，约 1GB，勿删）。
#   3) 接口: /v1/chat/completions  /v1/models  /health，模型 opendoc-0.1b / unirec-0.1b。
#   4) GPU 显存不足时 opendoc 会静默丢内容块（HTTP 仍 200），发布后必须查:
#        docker logs openocr-openai 2>&1 | grep -c "Error processing block"
#   5) 显存大头是 UniRec encoder 的自注意力（按输入边长四次方增长）。--max-pixel-side
#      在进模型前把最长边压小，不改 openocr 任何配置；越小越省显存，密集小字会掉精度。
# ============================================================================
set -euo pipefail

CONTAINER="${CONTAINER:-openocr-openai}"
IMAGE="${IMAGE:-openocr-openai:1.0}"
PORT="${PORT:-8089}"
MODELDIR="${MODEL_DIR:-/data/ocr/openocr-openai/models}"
USEGPU="false"
MAXPIXEL=""
CPUSET="${CPUSET:-}"
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)             USEGPU="true"; shift ;;
    --image)           IMAGE="openocr-openai:$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --max-pixel-side)  MAXPIXEL="$2"; shift 2 ;;
    --model-dir)       MODELDIR="$2"; shift 2 ;;
    --cpuset)          CPUSET="$2"; shift 2 ;;
    --dry-run)         DRY=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

CMD=(docker run -d --name "$CONTAINER"
  --gpus all
  -p "${PORT}:8089"
  -v "${MODELDIR}:/root/.cache/openocr"
  -e "OPENOCR_USE_GPU=${USEGPU}")

# 绑核：共享机器上容器默认会飘到 E-core（285K 的 8-23 号核明显慢于 0-7 的 P-core），
# 单图延迟随调度抖动。绑 P-core 后延迟稳定；server.py 已把 ORT 线程限到 8，
# 与 8 个 P-core 对齐，不会超订。
if [ -n "$CPUSET" ]; then
  CMD+=( --cpuset-cpus "${CPUSET}" )
fi

# 环境变量必须在镜像名之前，追加到镜像后面会被当成容器启动命令
if [ -n "$MAXPIXEL" ]; then
  CMD+=( -e "OPENOCR_MAX_PIXEL_SIDE=${MAXPIXEL}" )
fi

CMD+=( --restart unless-stopped "$IMAGE" )

if [ "$DRY" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
"${CMD[@]}"

# 模型加载需要时间，轮询健康检查最长 120s
for i in $(seq 1 24); do
  sleep 5
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "OK: ${CONTAINER} 已就绪 (port ${PORT}, GPU=${USEGPU}, image=${IMAGE}, max_pixel_side=${MAXPIXEL:-默认1024}, model_dir=${MODELDIR}, cpuset=${CPUSET:-未绑核})"
    curl -s -m 5 "http://127.0.0.1:${PORT}/health"; echo
    exit 0
  fi
done
echo "WARN: 120s 内未就绪，查看: docker logs ${CONTAINER}"
exit 1
