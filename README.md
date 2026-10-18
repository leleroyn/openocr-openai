# OpenOCR OpenAI 兼容接口服务

把 [OpenOCR](https://github.com/Topdu/OpenOCR) 的 **OpenDoc-0.1B**（整页文档解析 → Markdown）与
**UniRec-0.1B**（通用文字 / 公式 / 表格识别）封装成 **OpenAI Chat Completions 兼容** 的 HTTP 服务，
任何 OpenAI 客户端 / SDK / 网关（new-api、one-api、LiteLLM…）改一下 base_url 就能直接调用 OCR。

> 纯 OCR 模型：**只接受图片输入**（base64 data URI 或 http(s) 链接），支持单请求多图；
> 没有对话语义，不消耗 token 计费（`usage` 恒为 0）。

<p>
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-REST-blue">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-GPU%20%7C%20CPU-blue">
  <img alt="OpenAI compatible" src="https://img.shields.io/badge/API-OpenAI%20compatible-green">
</p>

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [接口](#接口)
- [调用示例](#调用示例)
- [接入 new-api / one-api 网关](#接入-newapi--oneapi-网关)
- [配置项](#配置项)
- [性能与显存](#性能与显存)
- [自检脚本](#自检脚本)
- [项目结构](#项目结构)
- [限制与注意事项](#限制与注意事项)
- [许可](#许可)

## 特性

- **OpenAI 协议原样兼容**：`POST /v1/chat/completions`、`GET /v1/models`，标准 `chat.completion` 响应体，官方 openai SDK 直接可用。
- **支持流式**：`stream=true` 走标准 SSE（`chat.completion.chunk` + 结尾 `data: [DONE]`）。
- **网关友好**：纯文本请求（网关的渠道连通性测试）返回 **200 + 能力说明文本**，不返回 400，渠道测试直接通过。
- **单请求多图**：多图按顺序逐张识别，结果按 `### 图片 i/N` 分段拼接。
- **前置缩放压显存**：进模型前把最长边压到上限并重编码 JPEG，**不改 OpenOCR / ONNX 任何配置**，显存随边长四次方下降。
- **GPU / CPU 一键切换**：`OPENOCR_USE_GPU=true` 走 CUDA，显存不够时一条命令回退 CPU。
- **单锁串行推理**：并发请求排队，避免显存/内存被打爆。
- **零依赖客户端**：curl / Python / Java / Node 任何能发 HTTP POST JSON 的语言都能用。

## 快速开始

### 1. Docker（推荐）

```bash
docker build -t openocr-openai:1.0 .

# CPU 模式（无需显卡，最稳）
docker run -d --name openocr-openai -p 8089:8089 \
  -v $PWD/models:/root/.cache/openocr \
  openocr-openai:1.0

# GPU 模式（需空闲显存 >= 4.5GB）
docker run -d --name openocr-openai --gpus all -p 8089:8089 \
  -v $PWD/models:/root/.cache/openocr \
  -e OPENOCR_USE_GPU=true \
  openocr-openai:1.0
```

`/root/.cache/openocr` 是 OpenOCR 的模型缓存目录（约 1GB，首次启动自动下载）。挂载出来可复用，
容器重建不用重新下载；GPU 模式下这份挂载是必需的（模型只加载一次）。

启动约需 1~2 分钟加载模型，就绪后：

```bash
curl -s http://127.0.0.1:8089/health
# {"status":"ok","models":["opendoc-0.1b","unirec-0.1b"],"loaded":[...],"gpu":false,"max_pixel_side":1024,"max_image_kb":5120}
```

### 2. 裸机

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py            # 监听 0.0.0.0:8089
```

GPU 环境请把 `requirements.txt` 里的 `onnxruntime` 换成 `onnxruntime-gpu`（Dockerfile 已处理）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | OCR 推理（OpenAI Chat Completions 兼容，支持 `stream`） |
| GET | `/v1/models` | 模型列表 |
| GET | `/health` | 健康检查（含已加载模型、GPU 开关、缩放上限） |

### 模型

| 模型名 | 能力 | 输入建议 |
|---|---|---|
| `opendoc-0.1b` | 整页文档解析 → Markdown（含版面检测、阅读顺序、表格） | A4 整页、截图、票据整页 |
| `unirec-0.1b` | 单行 / 单块文字识别，支持 LaTeX 公式与表格结构 | 裁剪后的文字条、公式、表格单元 |

### 请求

```json
{
  "model": "opendoc-0.1b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "任意说明文本（当前版本忽略）"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/page.png"}}
      ]
    }
  ],
  "max_tokens": 2048
}
```

- `model`：`opendoc-0.1b` 或 `unirec-0.1b`，其他值返回 404 `model_not_found`。
- 图片：`content` 数组里放 1~N 个 `image_url`，base64 data URI 与 http(s) 链接都支持，**按顺序逐张识别**。
- `max_tokens`：映射为 UniRec 的 `max_length`，默认 2048，自动夹到 `[1, 2048]`（传 0 或缺省都用 2048）。
- `stream`：`true` 返回 SSE；推理是同步完成后分块下发（512 字/块），不是边推理边吐字。
- 网关常见附加字段（`temperature` / `n` / `seed` / `user` / `response_format` …）全部接受并忽略。
- **请求里没有图片时返回 200 + 能力说明文本**（不推理、不报 400），这是为了让网关的连通性测试通过。

### 响应

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "opendoc-0.1b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "识别结果文本 / Markdown"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

多图时 content 按 `### 图片 i/N: 文件名` 分段，用 `---` 分隔拼接。`usage` 固定为 0，**不要用于计费**。

### 错误

OpenAI 风格错误体，HTTP 码 + `error.message`：

| 场景 | HTTP | type |
|---|---|---|
| 模型名不存在 | 404 | `model_not_found` |
| 图片解析/下载失败 | 400 | `invalid_request_error` |
| 推理异常 | 500 | `server_error` |

## 调用示例

### curl

```bash
b64=$(base64 -w0 page.png)
curl -s http://127.0.0.1:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"opendoc-0.1b\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$b64\"}}]}]}"
```

### Python（官方 openai SDK）

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8089/v1", api_key="dummy")
b64 = base64.b64encode(open("page.png", "rb").read()).decode()

r = client.chat.completions.create(
    model="opendoc-0.1b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "解析这张图"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
)
print(r.choices[0].message.content)
```

可运行版本见 `examples/openai_sdk_demo.py`（图片路径走参数，地址走 `OCR_BASE` 环境变量）。

### Java（OkHttp / RestTemplate / Hutool 均可，就是普通 HTTP POST JSON）

```java
String body = "{\"model\":\"opendoc-0.1b\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," + b64 + "\"}}]}]}";
// POST http://127.0.0.1:8089/v1/chat/completions, Content-Type: application/json
// 取 choices[0].message.content
```

## 接入 new-api / one-api 网关

1. 渠道类型选 **OpenAI**，API 基址填 `http://<host>:8089/v1`（网关自己拼 `/chat/completions`，基址必须带到 `/v1`）。
2. 模型列表填 `opendoc-0.1b,unirec-0.1b`，测试模型填 `opendoc-0.1b`。
3. 渠道测试发的是纯文本 → 服务端返回 **200 + 说明文本**，测试通过；图片必须放在 `content` 数组的 `image_url` 里，网关原样透传。
4. 支持 `stream=true`，SSE 格式标准，网关流式转发正常。
5. 渠道超时设 **>= 120s**：CPU 模式 opendoc 3 图 10~38s、unirec 3 图 15~40s（机器被其他服务挤占时更慢）。
6. `usage` 恒为 0 → 网关计费为 0；服务端不校验 `Authorization`，渠道密钥填任意值即可。

## 配置项

全部通过环境变量配置，无需改代码：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `OPENOCR_USE_GPU` | `false` | `true`/`1`/`yes`/`auto` = CUDA GPU；`false` = CPU |
| `OPENOCR_MAX_PIXEL_SIDE` | `1024` | 进模型前的最长边上限，超过就等比缩小并重编码 JPEG。**越小越省显存**，密集小字会掉精度 |
| `OPENOCR_MAX_IMAGE_KB` | `5120` | 单图字节上限（5MB），超过重编码 JPEG 直到达标（质量下限 55） |
| `OMP_NUM_THREADS` | `4` | CPU 推理线程数 |
| `PORT` | `8089` | `python server.py` 的监听端口 |

挂载：`-v <本地目录>:/root/.cache/openocr` = 模型缓存（约 1GB，首次自动下载，建议持久化）。

## 性能与显存

CPU 模式（24 核，`max_pixel_side=1024`）实测：

| 用例 | 耗时 |
|---|---|
| 单图（裁剪文字条） | 0.2~0.3s |
| 整页 A4（unirec） | ~5s/页 |
| 整页 A4（opendoc，含版面检测） | ~3s/页 |
| 双图请求 | 6~10s |

请求内部**串行**（单锁），并发请求排队，客户端超时建议 ≥ 60s/页。

GPU 显存构成（fp32，模型权重只有 933MB，稳态 3550 MiB）：

| 阶段 | 显存 | 说明 |
|---|---|---|
| CUDA context | 336 MiB | 固定成本，与模型/输入大小无关 |
| 加载 4 个 ONNX 会话 | 2218 MiB | 权重上卡 + ORT 加载开销（opendoc 布局+识别、unirec 编码+解码，各一套显存池） |
| 首次推理 | 3550 MiB | fp32 激活 + cuDNN workspace + 2048 token KV cache，arena 只增不还 |

大头是 UniRec encoder 的自注意力，显存随输入边长**四次方**增长 —— 所以 `OPENOCR_MAX_PIXEL_SIDE`
是最有效的显存旋钮。GPU 空闲显存不足时服务会以 CPU 模式运行（opendoc 在显存贴边时可能静默丢内容块，
发布后建议查 `docker logs openocr-openai 2>&1 | grep -c "Error processing block"`）。

显存调优、发布流程、模型缓存目录细节见 `docs/DEPLOY.md`。

## 自检脚本

```bash
export OCR_BASE=http://127.0.0.1:8089/v1     # Windows: $env:OCR_BASE="..."; 默认本机

python test_api.py page.png                 # 两个模型 × 单图/多图，打印耗时与识别结果
python test_newapi.py page.png              # 网关接入自检 8 项：纯文本/流式/数组text/未知模型/附加字段/图片×3
```

`test_newapi.py` 8 项全部 HTTP 200（流式项要求 `[DONE]OK`）即接入正常。

## 项目结构

```
.
├── server.py                  # FastAPI 服务（全部逻辑，约 300 行）
├── requirements.txt
├── Dockerfile                 # python:3.10-slim + onnxruntime-gpu
├── deploy-openocr.sh          # 服务器端容器发布脚本（重建 + 健康检查轮询）
├── test_api.py                # 接口冒烟测试
├── test_newapi.py             # 网关接入自检（8 项）
├── examples/
│   ├── openai_sdk_demo.py     # 官方 openai SDK 调用示例
│   └── curl.sh                # curl 调用示例
├── docs/
│   └── DEPLOY.md              # 显存调优 / 发布流程 / 模型缓存细节
├── LICENSE                    # Apache-2.0
└── NOTICE
```

## 限制与注意事项

- **仅图片输入**：PDF 需调用方先转图，每页一图放进同一请求即可。
- **无对话语义**：`text` part 会被忽略，没有 system prompt 概念，不做多轮上下文。
- **无鉴权**：服务不校验 `Authorization`，仅供内网使用；公网暴露请自行加网关 / token / IP 白名单。
- **无 function calling / tools / 多轮对话语义**，`usage` 恒为 0。
- 模型权重（OpenDoc-0.1B / UniRec-0.1B）有自己的许可条款，公开部署前请查看模型卡。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

本项目封装的 [OpenOCR](https://github.com/Topdu/OpenOCR) 及其模型权重遵循各自的许可条款。
