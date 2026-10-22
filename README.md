# ScriptMoE OpenAI 兼容 OCR 服务

把 [ScriptMoE](https://github.com/YesianRohn/ScriptMoE)（45.9M 参数的多语种 OCR 识别模型）封装成
**OpenAI Chat Completions 兼容** 的 HTTP 服务：任何 OpenAI 客户端 / SDK / 网关
（new-api、one-api、LiteLLM…）改一下 base_url 就能直接调用 OCR。

检测 + 识别两段式：

| 环节 | 模型 | 运行时 |
|---|---|---|
| 文本检测 | OpenOCR repvit_db（12MB ONNX） | ONNX Runtime（CPU / CUDA EP） |
| 文字识别 | ScriptMoE（SVTRv2 encoder + MoE decoder，47.8M 参数） | PyTorch（CPU / CUDA） |

输出是**按阅读顺序组装的纯文本**：带框的文字行先按 y 聚成文本行，行内按 x 拼接，行间换行。

> 纯 OCR 模型：只接受图片输入（base64 data URI 或 http(s) 链接），支持单请求多图；
> 没有对话语义，不消耗 token 计费（usage 恒为 0）。

<p>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
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
- [与 OpenDoc / UniRec 的差异](#与-opendoc--unirec-的差异)
- [输出防护](#输出防护)
- [自检脚本](#自检脚本)
- [项目结构](#项目结构)
- [限制与注意事项](#限制与注意事项)
- [许可](#许可)

## 特性

- **OpenAI 协议原样兼容**：`POST /v1/chat/completions`、`GET /v1/models`，标准 `chat.completion` 响应体，官方 openai SDK 直接可用。
- **支持流式**：`stream=true` 走标准 SSE（`chat.completion.chunk` + 结尾 `data: [DONE]`）。
- **网关友好**：纯文本请求（网关的渠道连通性测试）返回 **200 + 能力说明文本**，不返回 400，渠道测试直接通过。
- **单请求多图**：多图按顺序逐张识别，结果按 `### 图片 i/N` 分段拼接。
- **阅读顺序组装**：同一行上的「标签 + 值」拼成一行，行内按 x 排序，两侧都是中文时不插空格、中英文相邻插一个空格 —— 证照/表单按行抽字段最省事。
- **只有一个模型 id**：`scriptmoe`。`opendoc-0.1b` / `unirec-0.1b` 连同名别名一起删掉了（引擎已不存在，留着别名只会让渠道把请求喂给不存在的能力）；new-api 渠道的模型名要改成 `scriptmoe`。
- **GPU / CPU 一键切换**：`OPENOCR_USE_GPU=true` 走 CUDA，一条命令回退 CPU。
- **逐行重复截断**：ScriptMoE 是定长 100 步解码，一行被同一子串刷屏（`////`、`159595959…`）时截断，正常重复行（整表同内容的行）不受影响。
- **资源上限齐全**：请求体、远端图片体积与超时、识别 batch、行宽高比全部可配，不会被一个大请求拖死。
- **单锁串行推理**：并发请求排队，避免显存/内存被打爆。
- **CPU 线程上限**：ORT 与 torch 用同一个 `OPENOCR_CPU_THREADS`，共享机器上不会自旋线程抢核（实测限线程后延迟稳定）。
- **零依赖客户端**：curl / Python / Java / Node 任何能发 HTTP POST JSON 的语言都能用。
## 快速开始

### 0. 取权重（不进 git）

| 文件 | 大小 | 来源 |
|---|---|---|
| `model.safetensors` | 191MB | ScriptMoE 仓库 `E2EOCR/model.safetensors`（git-lfs） |
| `dict.txt` | 95KB | 同仓库 `E2EOCR/assets/dict.txt`（19683 字符 + 空格） |

放到 `models/scriptmoe/` 下即可（裸机）；Docker 挂载 `/app/models`（见下）。
检测权重 `openocr_det_model.onnx`（12MB）首次启动由 openocr 走 modelscope 自动下载并缓存。

### 1. Docker（推荐）

```bash
docker build -t openocr-openai:2.0 .
docker run -d --name openocr-openai \
  -p 8089:8089 \
  -v /data/ocr/openocr-openai/models:/app/models \
  -v /data/ocr/openocr-openai/models:/root/.cache/openocr \
  openocr-openai:2.0

curl -s http://127.0.0.1:8089/health
# {"status":"ok","models":["scriptmoe"],"loaded":["scriptmoe"],"gpu":false,"device":"cpu",
#  "det_mode":"mobile","max_ratio":20,"max_text_length":100,"rec_batch":8,"drop_score":0.5,
#  "cpu_threads":8,"max_image_kb":5120}
```

发布用 `deploy-openocr.sh`（本地构建镜像 → save 上传 → 服务器 load，脚本内建「镜像存在 + 权重存在」两道检查）：

```bash
bash deploy-openocr.sh --gpu --rec-batch 4 --cpuset 0-7
```

### 2. 裸机

```bash
pip install -r requirements.txt
# torch 要装对变体（PyPI 默认装到的是 CUDA 版，CPU 机器上白装 2GB+）：
#   CPU:          pip install torch --index-url https://download.pytorch.org/whl/cpu
#   GPU(GTX10系): pip install torch --index-url https://download.pytorch.org/whl/cu126
uvicorn server:app --host 0.0.0.0 --port 8089
```

国内网络取 torch CPU 轮子：`pip install "torch==2.14.1+cpu" --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/`。

## 接口

| 端点 | 说明 |
|---|---|
| `GET /health` | 就绪探针，返回引擎配置（模型加载完才 `ok`） |
| `GET /v1/models` | 只列规范 id：`scriptmoe` |
| `POST /v1/chat/completions` | OCR 主接口，支持 `stream`、多图、base64 与远端 URL |

### 模型

`scriptmoe` 是唯一模型 id，请求里的 `model` 必须精确等于它。
请求里写它们一样能出结果（响应里的 `model` 会规范化成 `scriptmoe`），网关渠道无需改动。

### 请求

```json
{
  "model": "scriptmoe",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "识别"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }],
  "stream": false
}
```

- 图片放在 `content` 数组的 `image_url` 里：base64 data URI 或 http(s) 链接都行，单请求多张。
- `max_tokens` 接受但不参与推理控制：ScriptMoE 是**逐行定长解码**，单行上限 `SCRIPTMOE_MAX_TEXT_LENGTH` 个字符，行数由检测框决定，没有全局 token 预算可以截断。
- 纯文本请求（无图片）返回 200 + 能力说明，供网关渠道测试用。

### 响应

标准 `chat.completion`，`content` 是纯文本。中国营业执照实测的输出形态：

```text
名称 襄阳 xx 管理有限公司
统一社会信用代码 91420600343516164T
注册资本 壹仟万圆整
类型 有限责任公司(自然人投资或控股)
成立日期 2011年06月28日
法定代表人 张 xx
地址 樊人民路 1 号
```

同行两个文字块（标签 + 值）拼成一行；两侧都是中文时不插空格，中英文相邻插一个空格。
识别置信度低于 `SCRIPTMOE_DROP_SCORE` 的行直接丢弃，不进 `content`。

### 错误

| HTTP | 场景 |
|---|---|
| 400 | 图片解析失败（base64 坏、URL 协议不支持、下载超时/超体积） |
| 404 | 模型 id 不等于 `scriptmoe`（旧 id `opendoc-0.1b` / `unirec-0.1b` 已删，不再有别名） |
| 413 | 请求体超过 `OPENOCR_MAX_BODY_MB` |
| 500 | 推理异常（详情在服务端日志，响应只带一句摘要） |
## 调用示例

### curl

```bash
B64=$(base64 -w0 图片.png)
curl -s http://127.0.0.1:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"scriptmoe\",\"messages\":[{\"role\":\"user\",\"content\":[
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}]}"
```

完整可跑的例子见 `examples/curl.sh`、`examples/openai_sdk_demo.py`。

### Python（官方 openai SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8089/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="scriptmoe",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/lic.jpg"}},
    ]}],
)
print(r.choices[0].message.content)
```

### Java（OkHttp / RestTemplate / Hutool 均可，就是普通 HTTP POST JSON）

`POST http://127.0.0.1:8089/v1/chat/completions`，`Content-Type: application/json`，
body 结构见[请求](#请求)；`Authorization` 头随便填（服务端不校验）。

## 接入 new-api / one-api 网关

1. 渠道类型选 `OpenAI`，API 地址填 `http://&lt;host&gt;:8089`（不带 `/v1`）。
2. 模型名填 `scriptmoe`（老渠道里原来的 `opendoc-0.1b` / `unirec-0.1b` 必须改掉，现在会 404）。
3. 渠道测试：网关发纯文本 `hi` → 本服务返回 **200 + 说明文本**，测试直接过。
4. 流式已支持，`[DONE]` 结尾正常。

`python test_newapi.py 图片.jpg` 会把纯文本 / 流式 / 旧 id 应 404 / 未知模型 / 附加字段 / 图片逐条打出来。

## 配置项

全部走环境变量，默认配置在一台普通机器上开箱可用，不需要设任何东西。

| 变量 | 默认 | 说明 |
|---|---|---|
| `ORT` | 8089 | 监听端口 |
| `OPENOCR_LOG_LEVEL` | INFO | 日志级别 |
| `OPENOCR_USE_GPU_` | false | true 走 CUDA（识别 torch 与检测 ONNX 都上 GPU） |
| `OPENOCR_CPU_THREADS` | CPU 模式 min(8, 核数) | 同时用于 ONNX Runtime 与 torch；0 = 用各自默认 |
| `SCRIPTMOE_WEIGHTS` | models/scriptmoe/model.safetensors | 识别权重路径 |
| `SCRIPTMOE_DICT` | models/scriptmoe/dict.txt | 字符字典路径 |
| `SCRIPTMOE_DET_MODEL` | 空（首次在线下载） | 检测 ONNX 路径，离线环境用 |
| `SCRIPTMOE_DET_MODE` | mobile | 检测器规格（repvit_db mobile） |
| `SCRIPTMOE_MAX_RATIO` | 20 | 行宽高比上限，**显存与耗时随它平方增长**，见[性能与显存](#性能与显存) |
| `SCRIPTMOE_MAX_TEXT_LENGTH` | 100 | 单行最多几个字符（decoder 步数上限） |
| `SCRIPTMOE_DROP_SCORE` | 0.5 | 低于该识别置信度的行直接丢 |
| `SCRIPTMOE_REC_BATCH` | 8 | 一次进多少个裁条（同宽高比才拼得进 batch） |
| `OPENOCR_TRIM_MIN_TIMES` | 8 | 单行结尾同一子串连续重复几次判定刷屏并截断 |
| `OPENOCR_MAX_PIXEL_SIDE` | 0（不缩放） | 只当解码炸弹的保险丝用：识别裁条反正压到 32 高，缩放只掉精度 |
| `OPENOCR_MAX_IMAGE_KB` | 5120 | 单图体积上限，超过重编码 JPEG |
| `OPENOCR_URL_TIMEOUT` | 30 | 远端图片下载超时（秒） |
| `OPENOCR_MAX_URL_MB` | 20 | 远端图片体积上限 |
| `OPENOCR_MAX_BODY_MB` | 64 | 请求体上限（base64 膨胀 4/3） |
## 性能与显存

### CPU（i5-7400 4C4T，4 线程，11 张中国营业执照）

| 引擎 | 平均 ms/张 | 关键字段命中（8 字段 x 11 张） | 18 位代码校验位通过 |
|---|---|---|---|
| **ScriptMoE（本服务，API 端到端）** | **17794** | **84/88 = 95%** | **11/11** |
| ScriptMoE（上游 E2EOCR 进程内跑，不含 HTTP 与 base64） | 16511（det 0.24–0.65s + rec 10–25s） | 84/88 = 95% | 11/11 |
| OpenOCR repsvtr_ch（ONNX mobile） | 3335 | 74/88 = 84% | 8/11 |
| indra-ocr（CPU ONNX） | 1411 | 66/88 = 75% | 7/11 |

四行数据同一台机器、同一批图、同一个检测器（repvit_db mobile），差别只在识别模型。
关键字段 = 代码 / 名称 / 类型 / 法定代表人 / 注册资本 / 成立日期 / 住所 / 经营范围 是否抽得到；
18 位代码校验位 = 用 GB32100 校验位规则（前 17 位按 3^i mod 31 加权）验识别出来的信用代码是不是合法编号：
错任意一个字符校验位几乎必然过不了，且不需要人工标注，是证照场景最硬的正确性指标。

这两列可以用仓库里的脚本在自己的证照集上复现：

```bash
python test_batch.py 图片目录 --out texts.json   # 走 API 跑一批，逐张打印耗时
python score_batch.py texts.json [texts2.json …] # 8 关键字段命中率 + GB32100 校验位，多份并列便于横向比
```

- 识别是自回归解码，耗时 ≈ 行数 x 每行最长文本，**CPU 上单张证照 15~25s 是正常水平**：
  适合「一张图几十秒、要准不要快」的场景，要秒级响应请上 GPU。
- 单锁串行：并发请求排队，QPS 约等于 1 / 单张耗时。
- 图片默认不缩放（`OPENOCR_MAX_PIXEL_SIDE` 默认 0）：识别裁条会被统一压到 32 高，缩放只在进检测器前丢像素。
  旧引擎 462 张营业执照实测关键字段命中 原图 92 / 2048px 85 / 1024px 74，同一个结论：设缩放上限只丢像素不省时间。

### GPU 显存预算

单图激活是 encoder 第三段全局注意力的 `(B, 8, N, N)` 开销，**随 `SCRIPTMOE_MAX_RATIO` 平方增长**
（N = (32/4) x (32 x ratio / 4)，ratio 20 时 N=1280，单图单算子 52.4 MiB）。CPU 实测，设备无关：

| ratio | 识别输入 | 单图激活 |
|---|---|---|
| 20 | 32 x 640 | 112 MiB |
| 8 | 32 x 256 | 26 MiB |

| rec_batch | ratio 20 激活合计 | ratio 8 激活合计 |
|---|---|---|
| 1 | 182 MiB | — |
| 4 | 474 MiB | — |
| 8 | 906 MiB | 205 MiB |
| 16 | 1770 MiB | — |
| 32 | 3464 MiB | 1560 MiB |

总预算 = 权重 175 MiB + CUDA 上下文 250~350 MiB + 激活 + 碎片 ~15%：

| 配置 | 稳态显存 | 适合 |
|---|---|---|
| batch 8 x ratio 20 | ≈ 1.4 GiB | 4GB 卡（1050 Ti 桌面已占 ~1GB，净 3.1GiB，够用） |
| batch 16 x ratio 20 | ≈ 2.6 GiB | 6GB 以上 |
| batch 32 x ratio 8 | ≈ 1.6 GiB | 窄条文字为主的场景，省一半显存 |

- **GTX 10 系（Pascal / sm_61）必须装 cu126 版 torch**：PyTorch 2.8 起的 cu128/cu129 wheel 已移除 Pascal 内核，CUDA 13.0 也弃用 Pascal。
- Pascal 没有 Tensor Core，GPU 的收益来自 batch 并行，不来自半精度。

## 与 OpenDoc / UniRec 的差异

本服务的前身是 OpenDoc-0.1B（整页文档 → Markdown）+ UniRec-0.1B（单块识别）。换引擎是**能力交换**，不是纯升级：

**得到的**（11 张营业执照实测）：

- 关键字段命中 84/88 = 95%（vs repsvtr_ch 74/88 = 84%）、18 位统一社会信用代码**校验位 11/11 全过**（旧引擎 8/11）；
- 多语种覆盖：词表 19683 字符（拉丁 / 西里尔 / 阿拉伯 / 天城文…），旧引擎主要中英；
- 输出里没有 LaTeX 化：旧引擎会把「每个字符一个方框」的代码行输出成 `$ 9~1~3~5~… $`，现在直接是字符。

**失去的**（调用方要提前知道）：

- **没有版面块、没有 HTML / Markdown 表格**：输出是纯文本行，表格不会变成 `<table>` 或 `|---|`；
- 没有公式（词表里没有 LaTeX 命令），印章 / 图片不再有占位标记；
- 没有整页文档的层级结构（标题块 / 段落块），调用方按行自己解析；
- CPU 上慢一个量级（16.5s vs 3.3s）。

**因此删掉的服务端代码**（都随旧引擎一起失效，删除后全仓引用计数为 0）：
版面检测 resize 取整补丁、UniRec 解码层循环早停补丁、LaTeX / 图片标签清洗、
表格骨架重复阈值、`MAX_TOKENS_CAP`、`MAX_PARALLEL_BLOCKS`、doc / unirec 引擎初始化与专属环境变量。
文本层的逐行重复截断保留（`_trim_repeats`）。

## 输出防护

ScriptMoE 的 decoder 每行最多解 100 步，整批 EOS 就早停，**不存在旧引擎那种跑满 2048 步的烧穿**。
剩下的故障形态是单行内容被同一子串刷屏（`////`、`159595959…`）：
`_trim_repeats` 找行尾最小重复子串，连续重复超过 `OPENOCR_TRIM_MIN_TIMES`（默认 8）次就截成 `…`。
截断是**逐行**做的，20 行内容完全相同的正常表不会被砍 —— 用例见 `test_engine.py`。

## 自检脚本

```bash
python test_engine.py                      # 纯 Python 单测，秒级，不加载模型
python test_api.py 图片.jpg [图片2.jpg …]    # scriptmoe × 单图 / 多图，打印耗时与结果
python test_newapi.py 图片.jpg              # 模拟网关渠道测试：纯文本 / 流式 / 旧 id 404 / 附加字段 / 图片
python test_batch.py 图片目录 --out t.json  # 批量跑一批图，逐张耗时，产出给打分脚本用
python score_batch.py t.json               # 关键字段命中率 + GB32100 校验位打分
```

`test_engine.py` 覆盖：阅读顺序组装（行聚组 / 行内排序 / 中英文分隔符）、逐行截断阈值、
检测框排序、模型 id 精确匹配（旧 id 不在列表且无别名对象）、前置缩放与 EXIF 转正，以及**预处理与 torchvision 的逐像素一致性**
—— 本项目刻意不装 torchvision（识别侧只要 torch），这条用例就是那个换法的验收线，
没装 torchvision 的环境（生产容器）自动跳过。

## 项目结构

```text
server.py              FastAPI 服务：OpenAI 兼容层 + 引擎装配 + 资源上限
ocr_engine/
  pipeline.py          检测 + 识别 + 阅读顺序组装（服务端适配层）
  modeling.py          ScriptMoE 网络结构（上游 vendor，勿改）
  postprocess.py       自回归解码后处理（上游 vendor，勿改）
  crops.py             文字框仿射裁条
models/scriptmoe/      权重，不进 git：model.safetensors + dict.txt
deploy-openocr.sh      发布脚本（镜像检查 / 权重检查 / 健康检查）
examples/              curl 与 openai SDK 示例
test_engine.py         引擎与输出层单测
test_api.py            接口冒烟
test_newapi.py         网关接入自检
test_batch.py          批量跑图（耗时统计）
score_batch.py         关键字段 + 校验位打分
```

## 限制与注意事项

- **只吃图片**：无图片请求返回的是能力说明（不是识别结果）；PDF 需要调用方先转成图片。
- **EXIF 方向**：检测 / 识别侧用 `cv2.imread` 读图，**不看 EXIF**。服务端在 `_shrink` 里按 EXIF 统一转正，
  自己接图片流时要做同样的事，否则手机拍的证照整页识别质量直接崩。
- **CPU 慢**：单张证照 15~25s 且串行排队，容量规划请当并发 1 来算。
- **检测决定上限**：检测丢框的行永远识别不出来；`drop_score` 调低会带进噪声行
  （实测有竖排单字被拆碎、水印串刷屏这类噪声，逐行截断能压掉一部分，压不掉的在输出里）。
- **图片体积上限是尽力而为**：不可压缩内容（纯噪声、密集小字截图）在最低质量 55 下仍可能超过
  `OPENOCR_MAX_IMAGE_KB`，这时不会无限循环压下去，只做有界努力。
- **单行 100 字上限**：`SCRIPTMOE_MAX_TEXT_LENGTH` 之外的字符会被截断（单行超长的整页标题栏需要调大它，显存随之涨）。
- **上游权重许可**：ScriptMoE 仓库没有 LICENSE 文件，公开部署前需向上游确认权重许可（见 `NOTICE`）。
- **首帧**：模型加载 + 预热完成后 `/health` 才 `ok`；Docker `start-period` 给了 180s。

## 许可

- 本项目（OpenAI 兼容封装、脚本与文档）：MIT，见 `LICENSE`。
- `ocr_engine/modeling.py`、`ocr_engine/postprocess.py`：ScriptMoE 上游原样 vendor，见 `NOTICE`。
- 运行时依赖 OpenOCR / openocr-python（Apache-2.0）、PyTorch、ONNX Runtime 等，各自按上游许可。
- 模型权重不在本仓库内分发，按其自身条款使用。
