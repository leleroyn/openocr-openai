#!/bin/bash
# ============================================================================
# 发布/重建 openocr-openai（ScriptMoE 检测+识别的 OpenAI 兼容接口）
#
# 用法:
#   bash deploy-openocr.sh                        # 重建容器（CPU 模式，默认）+ 健康检查
#   bash deploy-openocr.sh --gpu                  # GPU 模式（需空闲显存 >=2.5GB，torch 需 cu126）
#   bash deploy-openocr.sh --image 2.0            # 指定镜像 tag
#   bash deploy-openocr.sh --port 8090            # 换端口（试验用）
#   bash deploy-openocr.sh --weights-dir /data/ocr/openocr-openai/models
#   bash deploy-openocr.sh --det-model /app/models/openocr_det_model.onnx  # 离线时用本地检测权重
#   bash deploy-openocr.sh --rec-batch 4          # 识别 batch（GPU 显存不够就调小）
#   bash deploy-openocr.sh --max-ratio 8          # 行宽高比上限（显存随它平方增长，默认 20）
#   bash deploy-openocr.sh --cpuset 0-7           # 绑到 P-core（共享机推荐）
#   bash deploy-openocr.sh --dry-run              # 只打印 docker run 命令，不执行
#
# 说明:
#   1) 镜像在本地构建，docker save + 上传，生产服务器只 load，禁止在服务器构建。
#   2) 识别权重走 --weights-dir（挂到容器 /app/models，约 190MB，勿删）：
#        /app/models/scriptmoe/model.safetensors
#        /app/models/scriptmoe/dict.txt
#   3) 检测权重（repvit_db ONNX，12MB）首次启动由 openocr 走 modelscope 下载，缓存在
#      --model-dir（挂到容器 /root/.cache/openocr）。服务器不通外网时，把
#      openocr_det_model.onnx 放进 --weights-dir 再用 --det-model 指过去。
#   4) 接口: /v1/chat/completions  /v1/models  /health，模型 scriptmoe
#      （旧 id opendoc-0.1b / unirec-0.1b 作为别名照样能用，网关渠道不用改）。
#   5) 显存大头是识别 encoder 第三段的全局注意力，随 SCRIPTMOE_MAX_RATIO 平方增长：
#      权重 175MB + CUDA 上下文 250~350MB + 激活（batch8 x ratio20 ≈ 1.4GiB）+ 碎片 ~15%。
#      4GB 卡（1050 Ti，桌面已占 ~1GB）用 --rec-batch 8 --max-ratio 20 起得来，
#      紧张就 --rec-batch 4；ratio 降到 8 时单图激活 112MB -> 26MB。
#   6) CPU 模式没有显存上限，图片默认不缩放（server.py 的 MAX_PIXEL_SIDE 默认 0）：
#      识别裁条反正会被统一压到 32 高，缩放只掉精度不省时间，别为了"省资源"设它。
# ============================================================================
set -euo pipefail

CONTAINER="${CONTAINER:-openocr-openai}"
IMAGE="${IMAGE:-openocr-openai:2.0}"
PORT="${PORT:-8089}"
DETCACHE="${MODEL_DIR:-/data/ocr/openocr-openai/models}"
WEIGHTS="${WEIGHTS_DIR:-/data/ocr/openocr-openai/models}"
USEGPU="false"
DETMODEL=""
RECBATCH=""
MAXRATIO=""
CPUSET="${CPUSET:-}"
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)          USEGPU="true"; shift ;;
    --image)        IMAGE="openocr-openai:$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --model-dir)    DETCACHE="$2"; shift 2 ;;
    --weights-dir)  WEIGHTS="$2"; shift 2 ;;
    --det-model)    DETMODEL="$2"; shift 2 ;;
    --rec-batch)    RECBATCH="$2"; shift 2 ;;
    --max-ratio)    MAXRATIO="$2"; shift 2 ;;
    --cpuset)       CPUSET="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 权重缺失就当场停：容器起来后 /health 一直不 ready，排查成本比这条检查高得多
if [ ! -f "${WEIGHTS}/scriptmoe/model.safetensors" ] || [ ! -f "${WEIGHTS}/scriptmoe/dict.txt" ]; then
  echo "ERROR: ${WEIGHTS}/scriptmoe/ 下缺 model.safetensors 或 dict.txt，已中止，未删除现有容器。"
  echo "       从 https://github.com/YesianRohn/ScriptMoE 的 E2EOCR/ 取这两个文件放好。"
  exit 1
fi

CMD=(docker run -d --name "${CONTAINER}"
  --gpus all
  -p "${PORT}:8089"
  -v "${DETCACHE}:/root/.cache/openocr"
  -v "${WEIGHTS}:/app/models"
  -e "OPENOCR_USE_GPU=${USEGPU}")

# 绑核：共享机器上容器默认会飘到 E-core（285K 的 8-23 号核明显慢于 0-7 的 P-core），
# 单图延迟随调度抖动。绑 P-core 后延迟稳定；server.py 已把 ORT 与 torch 的线程限到同一个值，
# 与 8 个 P-core 对齐，不会超订。
if [ -n "${CPUSET}" ]; then
  CMD+=( --cpuset-cpus "${CPUSET}" )
fi

# 环境变量必须在镜像名之前，追加到镜像后面会被当成容器启动命令
[ -n "${DETMODEL}" ] && CMD+=( -e "SCRIPTMOE_DET_MODEL=${DETMODEL}" )
[ -n "${RECBATCH}" ] && CMD+=( -e "SCRIPTMOE_REC_BATCH=${RECBATCH}" )
[ -n "${MAXRATIO}" ] && CMD+=( -e "SCRIPTMOE_MAX_RATIO=${MAXRATIO}" )

CMD+=( --restart unless-stopped "${IMAGE}" )

if [ "${DRY}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# 先确认镜像在本机存在，再删旧容器。
# 2026-10-19 踩过：--image 指定了不存在的 tag，脚本先 rm 再 run，run 失败 = 服务直接中断。
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: 本机没有镜像 ${IMAGE}，已中止，未删除现有容器。"
  echo "       先 docker load，或用 --image 指定已存在的 tag（docker images | grep openocr）"
  exit 1
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
"${CMD[@]}"

# 模型加载需要时间，轮询健康检查最长 120s
for i in $(seq 1 24); do
  sleep 5
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "OK: ${CONTAINER} 已就绪 (port ${PORT}, GPU=${USEGPU}, image=${IMAGE}, weights=${WEIGHTS}, det_cache=${DETCACHE}, rec_batch=${RECBATCH:-默认8}, max_ratio=${MAXRATIO:-默认20}, det_model=${DETMODEL:-在线下载}, cpuset=${CPUSET:-未绑核})"
    curl -s -m 5 "http://127.0.0.1:${PORT}/health"; echo
    exit 0
  fi
done
echo "WARN: 120s 内未就绪，查看: docker logs ${CONTAINER}"
exit 1
