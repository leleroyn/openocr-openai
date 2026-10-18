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

## 解码层补丁（重复循环 + 拷贝优化）

`server.py` 在运行时给 `UniRecONNX.decode_step` 打补丁，三件事：

1. **重复循环早停**：找尾部最小重复周期 p（1~8），同一周期连续重复 ≥3 遍且累计 token 数过阈值 → 该步
   logits 的 argmax 强制成 EOS，库的贪心循环自然 break。库原版只有 EOS 一个出口，一旦循环会跑满
   `max_length`（2048 步），单请求分钟级，且全程占着服务端串行锁，整台机器停摆。
2. **表格骨架单独阈值**：重复片段去掉 HTML 标签后不含任何文字/数字（`<td></td>`、`|---|---|`、`|  |  |`）
   判定为纯表格骨架，阈值从 32 token 放宽到 96 token。表格输出天然周期化，不区分就会误伤
   空单元格多、行结构规整的表。含内容的重复（`<td>金额</td>` 循环）仍按 32 token 判。
3. **省掉每步的张量拷贝**：原版每步对整页 encoder 的 `cross_k/cross_v` 和每层 `past_key/past_value`
   无条件 `.astype(np.float32)`（dtype 已是 float32 也照拷）。整页 1024px 时 cross KV 几十 MB，
   一次生成几百步 = GB 级 memcpy。补丁按 dtype 判断后复用数组。

**截断显式标记**：unirec 输出跑满 `max_tokens`（被长度切断，不是模型自然收尾）时，content 末尾追加
`> ⚠️ 输出达到 max_tokens=… 上限，内容被截断…`，并打 WARNING 日志。表格/长文档最容易撞上这个上限，
调用方必须能看出自己拿到的是半张表。

运维要点：

- 启动日志确认：`UniRec 解码补丁已生效：重复循环早停 周期<=8 且连续重复>=32 token（表格骨架 96 token）`。
  出现 `UniRecONNX.decode_step 签名已变，跳过解码补丁` 的 WARNING 说明 openocr 升级后签名不匹配，
  此时防护与拷贝优化都不生效，需要按新版源码改 `_patch_unirec`。
- 触发防护时日志：`检测到重复输出，已在第 34 步强制结束（连续重复 34 token，周期 3，正文阈值 32 / 表格骨架阈值 96）`。
  大量出现说明输入质量差（空白块/超宽长条/小字），考虑调小 `OPENOCR_MAX_PIXEL_SIDE` 或让调用方裁掉空白。
- **表格被误判成循环**（日志有「检测到重复输出」且表格缺行）→ 调大 `OPENOCR_REPEAT_STRUCT_MIN_TOKENS`（160/256）；
  正文循环漏判 → 调小 `OPENOCR_REPEAT_MIN_TOKENS`（24）；彻底关闭 → `OPENOCR_REPEAT_GUARD=false`。
- 表格质量优先时把 `OPENOCR_MAX_PIXEL_SIDE` 从 1024 提到 1408（UniRec 原生上限 960×1408，超过才缩），
  密集小字/多列表格精度明显变好，代价是显存按边长四次方增长（1024→1408 约 4 倍），CPU 模式无显存代价只是变慢。
- 补丁是**类级**的：unirec 引擎和 opendoc 内部的 `vlm_recognizer` 是同一个类，一处覆盖两个引擎。
  状态存在 `threading.local()` 里，因为 doc 任务多线程共享同一个 ONNX 会话。
- 版本耦合：`openocr-python==0.1.5`。升级前先在测试机跑 `python test_repeat_guard.py` + 真实图片对比。

## 版面块并行

opendoc 内部按版面块并行识别（`_parallel_vlm_recognize`，共享同一个 ONNX 会话，ORT 会话线程安全）：

| 环境 | `OPENOCR_MAX_PARALLEL_BLOCKS` | 原因 |
|---|---|---|
| CPU | `4`（默认） | `session.run` 释放 GIL，并行是真提速 |
| GPU | `1` | 并行几乎不涨吞吐，只抬高解码峰值显存 |

注意每个 worker 会多占一份解码工作内存，CPU 机器内存紧张时降到 2。

## 网关接入排障

| 现象 | 原因 / 处理 |
|---|---|
| 渠道测试报 `400 请求中没有图片` | 网关连通性测试只发纯文本。服务端 1.8 起对无图片请求返回 **200 + 能力说明文本**（不推理），收到该文本即代表连通正常 |
| 图片请求 400 | 图片必须放在 `content` 数组的 `image_url` 里（字符串 `content` 不带图片） |
| 流式无输出 | 确认 `stream=true`，SSE 结尾必须有 `data: [DONE]`；推理是同步完成后分块下发（512 字/块） |
| 网关超时 | 渠道超时设 >= 120s；CPU 模式 3 图 10~40s，机器被挤占时更慢 |
| GPU 模式 opendoc 结果缺内容块，HTTP 仍 200 | 显存贴边导致静默丢块。查 `docker logs openocr-openai 2>&1 | grep -c "Error processing block"`，降 `--max-pixel-side` 或回退 CPU |
| 单个请求卡住几十秒到几分钟 | 十有八九是模型陷入重复循环跑满 2048 步。确认启动日志里补丁已生效；防护生效时正文循环最坏 34 步、表格骨架 97 步 |
| 调用方拿到半张表 | 输出跑满 `max_tokens` 已自动加截断警告行；解法是按行块切图分多次请求，服务端上限 2048 |
| 表格缺行且日志有「检测到重复输出」 | 表格骨架被误判成循环，调大 `OPENOCR_REPEAT_STRUCT_MIN_TOKENS` |
| 远端图片 URL 卡住 | 已加 `OPENOCR_URL_TIMEOUT`（默认 30s）与 `OPENOCR_MAX_URL_MB`（默认 20MB）；库原版 `urlretrieve` 无超时，会把 worker 线程永久占住 |
| 大请求把内存打满 | `OPENOCR_MAX_BODY_MB`（默认 64MB）超限返回 413 |
| 计费异常 | `usage` 恒为 0，网关计费为 0；服务端不校验密钥，渠道 key 填任意值 |

自检：`python3 test_newapi.py <图片>`，8 项全 200（流式项 `[DONE]OK`）即接入正常。

## 版本要点

- **1.2（本仓库）**：解码层补丁（周期重复早停 + 表格骨架单独阈值 + 去掉每步 astype 拷贝）、
  跑满 `max_tokens` 显式截断标记、`max_parallel_blocks` 按 GPU/CPU 定默认、
  远端图片下载超时与体积上限、请求体上限、`_shrink` 之外全链路耗时日志、Windows 下 stdout 强制 UTF-8
  （openocr 日志带 emoji，GBK 环境会 `UnicodeEncodeError` 直接退出）。
- **1.8**：接入网关的兼容性修复 —— 无图片请求返回 200 说明文本（之前 400 会让渠道测试失败）；
  支持 `stream=true` 标准 SSE；`max_tokens` 自动夹到 `[1, 2048]`（避免 `max_length=0` 让 unirec 报错）。
  只影响无图片与流式两条路径，推理逻辑未变。
- **1.7 及之前**：前置缩放（`OPENOCR_MAX_PIXEL_SIDE` / `OPENOCR_MAX_IMAGE_KB`）、ORT 显存池参数、单锁串行。
