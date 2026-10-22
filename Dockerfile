# 构建参数：PIP_INDEX_URL 可换源（默认清华源，境外构建传 --build-arg PIP_INDEX_URL= 留空即可）
# TORCH_INDEX_URL 决定 torch 变体：
#   默认 cpu（CPU 推理，共享机器推荐）
#   GPU 构建传 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   （GTX 10 系 Pascal/sm_61 只能 cu126：PyTorch 2.8 起的 cu128/cu129 wheel 已移除 Pascal 内核）
# 国内构建 torch 用镜像：--build-arg TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu/
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
ARG PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_TRUSTED_HOST=download.pytorch.org

FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG TORCH_INDEX_URL
ARG TORCH_TRUSTED_HOST
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    TORCH_INDEX_URL=${TORCH_INDEX_URL} TORCH_TRUSTED_HOST=${TORCH_TRUSTED_HOST}

# libgl1 / libglib2.0-0：opencv（openocr 依赖）在 slim 镜像里的必需库
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch 从独立索引装：PyPI 默认的 torch（Windows 尤其）是 CUDA 版，CPU 机器上白装 2GB+。
# 所以构建时把 requirements.txt 里的 torch 行剔除，其余依赖走 PIP_INDEX_URL，torch 走 TORCH_INDEX_URL。
# 先装 CPU 版 onnxruntime（openocr 的依赖解析需要），再强制换成 GPU 版（CUDA EP 需要）。
RUN grep -v '^torch' requirements.txt > /tmp/req-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/req-no-torch.txt \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" --extra-index-url "${PIP_INDEX_URL}" \
         --trusted-host "${TORCH_TRUSTED_HOST}" torch \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.22.0

# pip 装进来的 CUDA 12 运行库需要显式加入动态库搜索路径
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib

COPY server.py .
COPY ocr_engine ./ocr_engine

# 权重不进镜像（191MB safetensors），运行时挂载 -v /data/ocr/openocr-openai/models:/app/models
# 目录结构：/app/models/scriptmoe/model.safetensors + /app/models/scriptmoe/dict.txt
ENV SCRIPTMOE_WEIGHTS=/app/models/scriptmoe/model.safetensors \
    SCRIPTMOE_DICT=/app/models/scriptmoe/dict.txt

EXPOSE 8089
# 模型加载 + 首次推理预热要 1 分钟量级，start-period 给足
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(\"http://127.0.0.1:\"+os.environ.get(\"PORT\",\"8089\")+\"/health\",timeout=5)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8089"]