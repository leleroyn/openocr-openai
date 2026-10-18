# 部署与显存调优

面向自托管场景（内网 GPU 机器 / 网关后端）。接口用法见 [../README.md](../README.md)。

## 部署原则

- **生产服务器禁止在线构建镜像。** 镜像在本地构建，`docker save` 成 tar 上传，服务器只做 `docker load` + 运行，避免构建抢生产资源。
- 运行文件统一放一个目录（例：`/data/ocr/openocr-openai/`），发布脚本单独放（例：`/data/ocr/scripts/`）。
- 服务器上不放构建文件（`Dockerfile` / `requirements.txt` 只留本地）。
- 模型缓存目录必须持久化挂载，容器重建不重复下载（约 1GB）。

## 服务器目录结构（参考）

```
/data/ocr/openocr-openai/
├── server.py            # 源码留档（镜像内含，服务器上无需构建）
├── test_api.py          # 接口测试脚本
├── test_newapi.py       # 网关接入自检（8 项）
└── models/              # 挂载进容器 /root/.cache/openocr（约 1GB，勿删）
    ├── PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx
    └── unirec_0_1b_onnx/{unirec_encoder,unirec_decoder}.onnx + unirec_tokenizer_mapping.json
/data/ocr/scripts/deploy-openocr.sh
```

`models/` 下这两个**平铺目录**是 OpenOCR 运行时真正加载的路径（日志：
`Loading tokenizer from mapping file: /root/.cache/openocr/unirec_0_1b_onnx/...`）。

OpenOCR 缺文件时会调 `snapshot_download(cache_dir=~/.cache/openocr)`，那会在
`models/models/topdktu--*/snapshots/master/` 生成一份**完全相同的副本**（md5 一致，约 900MB）
再拷进平铺目录。这份缓存副本可以删；以后若触发过下载，清一次：

```bash
rm -rf models/models models/.lock
```

## 发布脚本

`deploy-openocr.sh`：`docker rm -f` 旧容器 → `docker run` 新容器 → 轮询 `/health` 最长 120s → 打印健康 JSON。

```bash
bash deploy-openocr.sh                        # 重建容器（CPU 模式，默认）+ 健康检查
bash deploy-openocr.sh --gpu                  # GPU 模式（需空闲显存 >= 4.5GB）
bash deploy-openocr.sh --image 1.8            # 指定镜像 tag
bash deploy-openocr.sh --max-pixel-side 832   # 前置缩放上限（默认 1024，越小越省显存）
bash deploy-openocr.sh --port 8090            # 换端口（试验用）
bash deploy-openocr.sh --model-dir /data/ocr/openocr-openai/models
bash deploy-openocr.sh --dry-run              # 只打印 docker run 命令，不执行
```

GPU/CPU、缩放上限都只由参数决定，与镜像无关 —— 显存紧张时一条命令即可回退 CPU。

镜像构建/上传（本地）：

```bash
docker build -t openocr-openai:1.8 .
docker save openocr-openai:1.8 -o openocr-openai-1.8.tar
# sftp / scp 上传到服务器，然后：
docker load -i openocr-openai-1.8.tar
```

## 前置缩放（省显存 + 提速，不改模型配置）

服务在把图片送进模型前检查：**最长边 > `OPENOCR_MAX_PIXEL_SIDE`（默认 1024）或文件 > `OPENOCR_MAX_IMAGE_KB`（默认 5MB）**
就等比缩小并重编码 JPEG（质量从 92 递减，下限 55）。

UniRec 的 `SimpleImageProcessor` 只做"超过 960×1408 才缩小"，输入小于该尺寸时原样使用，
所以前置压边长有效，且 OpenOCR / ONNX 的参数一个都不动。

整页 A4 实测（GPU）：

| 上限 | 实际尺寸 | 稳态显存 | 结果 |
|---|---|---|---|
| 1024 | ≤1024px | ~3.5GB | 正常 |
| 832 | ≤832px | ~2.6GB | 密集小字开始掉精度 |
| 1408 | ≤1408px | >4.5GB | 精度最好，显存吃紧时 OOM / 静默丢块 |

调参顺序建议：先降 `--max-pixel-side`（1024 → 832），仍不够再回退 CPU。

## 显存构成（实测拆解，模型权重只有 933MB）

逐阶段采样（`nvidia-smi` 每 0.2s 打进程显存）：

| 阶段 | 显存 | 增量 | 内容 |
|---|---|---|---|
| 进程建立 CUDA context | 336 MiB | +336 | CUDA driver + cuBLAS/cuDNN/cuBLASLt。**固定成本，与模型大小、输入大小都无关** |
| 加载 4 个 ONNX 会话 | 2218 MiB | +1882 | 权重 933 MiB 上卡 + ORT 加载开销 ~950 MiB（host→device 暂存、每会话 arena 起始块、算子元数据） |
| 首次推理 | 3550 MiB | +1332 | fp32 激活 + cuDNN workspace + 2048 token KV cache |
| **稳态** | **3550 MiB** | — | arena 只增不还，之后不再增长 |

权重：`unirec_decoder.onnx` 554MB + `PP-DoclayoutV2.onnx` 214MB + `unirec_encoder.onnx` 165MB = 933MB
（fp32，0.1B 参数 × 4 字节）。

三个结构性大头：

1. **CUDA context 336 MiB** —— 固定税。
2. **4 个独立会话各一套显存池** —— opendoc（布局检测 + 块识别）+ unirec（encoder + decoder），互不共享。
3. **UniRec encoder 自注意力** —— 显存随输入边长**四次方**增长，`OPENOCR_MAX_PIXEL_SIDE` 是最有效的旋钮。

服务端已做的显存优化（`server.py`）：

- 包装 `onnxruntime.InferenceSession`，对 CUDA EP 设 `arena_extend_strategy=kSameAsRequested`、
  `cudnn_conv_use_max_workspace=0`，并关掉 `enable_mem_pattern` / `enable_cpu_mem_arena`。
  默认配置下 4 个会话的 arena 只增不还，瞬时峰值会永久抬高上限（实测 3.7~4.2GB）。
  ⚠️ ORT 1.22 已移除 `enable_mem_arena` / `device_mem_limit`，写进 `provider_options` 会让 CUDA EP
  注册失败并**静默回退 CPU**。
- 前置缩放（见上）。
- 单锁串行推理，防止并发把显存/内存打爆。

## 网关接入排障

| 现象 | 原因 / 处理 |
|---|---|
| 渠道测试报 `400 请求中没有图片` | 网关连通性测试只发纯文本。服务端 1.8 起对无图片请求返回 **200 + 能力说明文本**（不推理），收到该文本即代表连通正常 |
| 图片请求 400 | 图片必须放在 `content` 数组的 `image_url` 里（字符串 `content` 不带图片） |
| 流式无输出 | 确认 `stream=true`，SSE 结尾必须有 `data: [DONE]`；推理是同步完成后分块下发（512 字/块） |
| 网关超时 | 渠道超时设 >= 120s；CPU 模式 3 图 10~40s，机器被挤占时更慢 |
| GPU 模式 opendoc 结果缺内容块，HTTP 仍 200 | 显存贴边导致静默丢块。查 `docker logs openocr-openai 2>&1 | grep -c "Error processing block"`，降 `--max-pixel-side` 或回退 CPU |
| 计费异常 | `usage` 恒为 0，网关计费为 0；服务端不校验密钥，渠道 key 填任意值 |

自检：`python3 test_newapi.py <图片>`，8 项全 200（流式项 `[DONE]OK`）即接入正常。

## 版本要点

- **1.8**：接入网关的兼容性修复 —— 无图片请求返回 200 说明文本（之前 400 会让渠道测试失败）；
  支持 `stream=true` 标准 SSE；`max_tokens` 自动夹到 `[1, 2048]`（避免 `max_length=0` 让 unirec 报错）。
  只影响无图片与流式两条路径，推理逻辑未变。
- **1.7 及之前**：前置缩放（`OPENOCR_MAX_PIXEL_SIDE` / `OPENOCR_MAX_IMAGE_KB`）、ORT 显存池参数、单锁串行。
