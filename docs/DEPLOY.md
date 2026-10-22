# 部署与显存调优

面向自托管场景（内网 GPU 机器 / 网关后端）。接口用法见 [../README.md](../README.md)。

## 部署原则

- **生产服务器禁止在线构建镜像。** 镜像在本地构建，`docker save` 成 tar 上传，服务器只做 `docker load` + 运行，避免构建抢生产资源。
- 运行文件统一放一个目录（例：`/data/ocr/openocr-openai/`），发布脚本单独放（例：`/data/ocr/scripts/`）。
- 服务器上不放构建文件（`Dockerfile` / `requirements.txt` 只留本地）。
- 权重目录必须持久化挂载，容器重建不重复下载。

## 服务器目录结构（参考）

```
/data/ocr/openocr-openai/
├── server.py                     # 源码留档（镜像内含，服务器上无需构建）
├── test_api.py                   # 接口冒烟（规范 id + 旧 id 别名）
├── test_newapi.py                # 网关接入自检（9 项）
└── models/                       # 挂到容器 /app/models
    ├── scriptmoe/
    │   ├── model.safetensors     # 191 MB 识别权重（fp32，0.1B）
    │   └── dict.txt              # 19683 字符词表
    └── openocr_det_model.onnx    # 12 MB 检测权重（离线部署时用）
/data/ocr/scripts/deploy-openocr.sh
```

`models/` 同时挂进容器的 `/app/models`（识别权重）与 `/root/.cache/openocr`（检测器缓存），一个目录两种用途，`deploy-openocr.sh` 里对应 `--weights-dir` 与 `--model-dir`。

两个权重的来源：

- 识别：`https://github.com/YesianRohn/ScriptMoE` 的 `E2EOCR/`（`model.safetensors` + `dict.txt`），放 `models/scriptmoe/`。
- 检测：repvit_db ONNX（12MB），首次启动由 openocr 调 `snapshot_download(cache_dir=~/.cache/openocr)` 走 modelscope 下载。
  服务器不通外网时，把 `openocr_det_model.onnx` 放进 `models/` 再用 `--det-model` 指过去，之后不再联网。

从 1.x（OpenDoc + UniRec）升上来的机器，旧权重可以整删，省 ~1GB：

```bash
rm -rf models/PP_DoclayoutV2_onnx models/unirec_0_1b_onnx models/models models/.lock
```

`models/models` 是 openocr 触发过在线下载时生成的重复副本（与平铺目录 md5 一致），可以删；以后触发过下载再清一次即可。

## 发布脚本

`deploy-openocr.sh`：先查权重 → 再查本机镜像 → 才 `docker rm -f` 旧容器 → `docker run` → 轮询 `/health` 最长 120s。
权重缺 / 镜像缺都当场中止，**不会先把服务删掉**（2026-10-19 踩过：`--image` 写错 tag，脚本先 rm 再 run，run 失败 = 服务直接中断）。

```bash
bash deploy-openocr.sh                        # CPU 模式（默认）+ 健康检查
bash deploy-openocr.sh --gpu                  # GPU 模式（torch 必须是 cu126，见下）
bash deploy-openocr.sh --image 2.0            # 指定镜像 tag
bash deploy-openocr.sh --port 8090            # 换端口（试验用）
bash deploy-openocr.sh --weights-dir /data/ocr/openocr-openai/models
bash deploy-openocr.sh --det-model /app/models/openocr_det_model.onnx   # 离线用本地检测权重
bash deploy-openocr.sh --rec-batch 4          # 识别 batch（显存不够就调小）
bash deploy-openocr.sh --max-ratio 8          # 行宽高比上限（显存随它平方增长，默认 20）
bash deploy-openocr.sh --cpuset 0-7           # 绑 P-core（共享机推荐）
bash deploy-openocr.sh --dry-run              # 只打印 docker run 命令，不执行
```

GPU/CPU、batch、ratio 全由参数决定，与镜像无关 —— 显存紧张时一条命令即可回退 CPU 或调小 ratio。
绑核有意义：共享机器上容器默认会飘到 E-core，单图延迟随调度抖动；`server.py` 已把 torch 与 ONNX Runtime 的线程数限到同一个值（`OPENOCR_CPU_THREADS`），与绑核数对齐不会超订。

镜像构建/上传（本地）：

```bash
# CPU 版（默认）
docker build -t openocr-openai:2.0 .
# GPU 版：Pascal 卡必须 cu126（见下），用构建参数换 torch 源
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t openocr-openai:2.0-gpu .
docker save openocr-openai:2.0 -o openocr-openai-2.0.tar
# scp 上传到服务器，然后：
docker load -i openocr-openai-2.0.tar
```

`requirements.txt` 里的 `torch` 行在构建时被 `grep -v '^torch'` 剔除，由 `TORCH_INDEX_URL` 决定装哪个变体，避免 CPU 镜像里塞进 2.5GB 的 CUDA 运行库。

## 显存预算（GPU）

识别 encoder 第三段全局注意力的 `(B, 8, N, N)` 激活是显存大头，**随 `SCRIPTMOE_MAX_RATIO` 平方增长**
（N = (32/4) x (32 x ratio / 4)；ratio 20 时 N=1280，单图单算子 52.4 MiB）。尺寸在 CPU 上实测，与设备无关：

| ratio | 识别输入 | 单图激活 |
|---|---|---|
| 20（默认） | 32 x 640 | 112 MiB |
| 8 | 32 x 256 | 26 MiB |

| rec_batch | ratio 20 激活合计 | ratio 8 激活合计 |
|---|---|---|
| 1 | 182 MiB | — |
| 4 | 474 MiB | — |
| 8（默认） | 906 MiB | 205 MiB |
| 16 | 1770 MiB | — |
| 32 | 3464 MiB | 1560 MiB |

总预算 = 权重 175 MiB + CUDA context 250~350 MiB + 激活 + 碎片 ~15%：

| 配置 | 稳态显存 | 适合 |
|---|---|---|
| batch 8 x ratio 20 | ≈ 1.4 GiB | 4GB 卡（1050 Ti 桌面已占 ~1GB，净显存约 3.1GiB，够用） |
| batch 16 x ratio 20 | ≈ 2.6 GiB | 6GB 以上 |
| batch 32 x ratio 8 | ≈ 1.6 GiB | 窄条文字为主的场景，省一半显存 |

- **GTX 10 系（Pascal / sm_61）必须装 cu126 版 torch**：PyTorch 2.8 起的 cu128/cu129 wheel 已移除 Pascal 内核，CUDA 13 也弃用 Pascal。装错版本会静默回退 CPU，`/health` 的 `device` 字段是判据。
- Pascal 没有 Tensor Core，GPU 收益来自 batch 并行，不来自半精度：不要为了省显存上 fp16，这代卡上它不省时间。
- 服务端给 ONNX Runtime 的 CUDA EP 设 `arena_extend_strategy=kSameAsRequested`、`cudnn_conv_use_max_workspace=0`，关掉 `enable_mem_pattern` / `enable_cpu_mem_arena`，防止 arena 只增不还把瞬时峰值变成永久上限。
  ⚠️ ORT 1.22 已移除 `enable_mem_arena` / `device_mem_limit`，写进 `provider_options` 会让 CUDA EP 注册失败并**静默回退 CPU**。
- 单锁串行推理：并发请求排队，显存峰值 = 单请求峰值，不会因并发叠加打爆显存。

## CPU 部署

- 单张证照 15~25s（i5-7400 4C4T 4 线程，11 张营业执照经 API 端到端实测平均 17.8s），QPS ≈ 1 / 单张耗时。这是自回归解码的固有代价，不是配置问题：要秒级响应请上 GPU。
- `OPENOCR_CPU_THREADS` 同时限制 torch 与 ONNX Runtime 线程，建议 = 物理核数；配合 `--cpuset` 绑 P-core。
- **不要为了省资源设 `OPENOCR_MAX_PIXEL_SIDE`**（默认 0 = 不缩放）：识别裁条会被统一压到 32 高，缩放只在进检测器前丢像素。旧引擎 462 张实测关键字段命中 原图 92 / 2048px 85 / 1024px 74，结论一致：缩放只丢像素不省时间。只有单图体积超限才有 `OPENOCR_MAX_IMAGE_KB`（默认 5MB）兜底。
- 内存侧无需特别限制：单请求峰值约等于 batch 激活（ratio 20 x batch 8 ≈ 900 MiB 张量）+ torch 运行时，8GB 机器够用。

## 网关接入排障

| 现象 | 原因 / 处理 |
|---|---|
| 渠道测试报 `400 请求中没有图片` | 网关连通性测试只发纯文本。服务端对无图片请求返回 **200 + 能力说明文本**（不推理），收到即代表连通正常 |
| 图片请求 400 | 图片必须放在 `content` 数组的 `image_url` 里（字符串 `content` 不带图片）；base64 与 http(s) URL 两种都支持 |
| 换模型 id 后渠道全 404 | 规范 id 是 `scriptmoe`；旧 id `opendoc-0.1b` / `unirec-0.1b` / `scriptmoe-0.1b` 作为别名照样 200，网关侧**不需要改**。`/v1/models` 只列规范 id |
| 期望 GPU 但 `/health` 的 `device` 是 `cpu` | torch 没拿到 GPU：查 `nvidia-smi -L`、`--gpus all`、torch 是否为 cu126（Pascal） |
| 流式无输出 | 确认 `stream=true`，SSE 结尾必须有 `data: [DONE]`；推理同步完成后分块下发 |
| 网关超时 | 渠道超时设 >= 120s（CPU 模式单图 15~25s，多图叠加，机器被挤占时更慢） |
| 某一行被同一子串刷屏（`////`、`159595959…`） | 自回归 decoder 的行内循环。`_trim_repeats` 找行尾最小重复子串，连续超过 `OPENOCR_TRIM_MIN_TIMES`（默认 8）次截成 `…`。大量出现说明该图质量差（超宽长条 / 小字）：调小 `SCRIPTMOE_MAX_RATIO` 到 8，或让调用方裁图 |
| 长行结尾少了字 | 每行最多解 `SCRIPTMOE_MAX_TEXT_LENGTH`（默认 100）步，按行定长解码，**没有全局 token 预算**，`max_tokens` 参数不起作用 |
| 表格输出成纯文本行 | 引擎能力边界：ScriptMoE 不输出 HTML / Markdown 表格、不输出版面块、不输出公式（词表无 LaTeX 命令）。要表格结构由调用方按行解析，或保留旧的 OpenDoc 服务 |
| 远端图片 URL 卡住 | `OPENOCR_URL_TIMEOUT`（默认 30s）、`OPENOCR_MAX_URL_MB`（默认 20MB）；库原版 `urlretrieve` 无超时，会把 worker 线程永久占住 |
| 大请求把内存打满 | `OPENOCR_MAX_BODY_MB`（默认 64MB）超限返回 413 |
| 计费异常 | `usage` 恒为 0，网关计费为 0；服务端不校验密钥，渠道 key 填任意值 |

自检（三层，逐层往上跑）：

```bash
python3 test_engine.py                 # 引擎层 12 项：聚行 / 排序 / 分隔符 / 截断阈值 / 权重 / 预处理一致性
python3 test_api.py <图> <图2>         # 接口层：规范 id + 旧 id 别名 x 单图 / 多图，打印耗时
python3 test_newapi.py <图>            # 网关层 9 项：纯文本 / 流式 / 别名 / 未知模型 / 附加字段 / 图片；全 200（流式 [DONE]OK）即接入正常
```

`test_engine.py` 的预处理一致性项是**回归护栏**：它把服务端 PIL 预处理与上游 torchvision 逐像素比对，偏差必须为 0。
改动 `ocr_engine/pipeline.py` 的预处理（resize / 归一化 / 通道序）后必跑，偏一个数识别就会整片退化。

## 版本要点

- **2.0（本仓库）**：识别引擎换成 ScriptMoE（多语种 0.1B，自研 vendor 的 `ocr_engine/`，不依赖 torchvision），
  OpenDoc-0.1B / UniRec-0.1B 及全部专属补丁（版面 resize 取整、解码层循环早停、表格骨架单独阈值、LaTeX 清洗、
  `MAX_TOKENS_CAP`、`MAX_PARALLEL_BLOCKS`）随之删除，全仓引用计数为 0。模型 id 统一为 `scriptmoe`（旧 id 留别名）。
  显存旋钮从 `OPENOCR_MAX_PIXEL_SIDE` 改为 `SCRIPTMOE_MAX_RATIO` x `SCRIPTMOE_REC_BATCH`。
- **1.8**：网关兼容修复 —— 无图片请求返回 200 说明文本（之前 400 会让渠道测试失败）；标准 SSE 流式；`max_tokens` 夹到 `[1, 2048]`。
- **1.2**：解码层补丁、截断显式标记、远端下载超时与体积上限、请求体上限、Windows 下 stdout 强制 UTF-8（openocr 日志带 emoji，GBK 环境会 `UnicodeEncodeError` 直接退出）。
- **1.7 及之前**：前置缩放、ORT 显存池参数、单锁串行。
