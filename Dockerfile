# 构建参数：PIP_INDEX_URL 可换源（默认清华源，境外构建传 --build-arg PIP_INDEX_URL= 留空即可）
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
ARG PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn

FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# libgl1 / libglib2.0-0：opencv（openocr 依赖）在 slim 镜像里的必需库
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# 先装 CPU 版 onnxruntime（openocr 的依赖解析需要），再强制换成 GPU 版
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.22.0

# pip 装进来的 CUDA 12 运行库需要显式加入动态库搜索路径
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib

COPY server.py .
EXPOSE 8089
# 模型加载要 1~2 分钟，start-period 给足
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(\"http://127.0.0.1:\"+os.environ.get(\"PORT\",\"8089\")+\"/health\",timeout=5)" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8089"]
