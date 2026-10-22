"""OpenAI 兼容的 ScriptMoE OCR 服务：只接受图片输入，支持单请求多图。

接入 new-api 等 OpenAI 兼容网关的设计要点：
- 纯文本请求（网关的渠道连通性测试）返回 200 + 说明文本，不返回 400，否则渠道测试过不了。
- 支持 stream=true，按 chat.completion.chunk 吐 SSE，结尾 data: [DONE]。
- 图片走 content 数组里的 image_url（base64 data URI 或 http(s)），单请求多图。

引擎 = OpenOCR repvit_db 文本检测（ONNX Runtime）+ ScriptMoE 识别（PyTorch），
输出是按阅读顺序组装的纯文本（见 ocr_engine/pipeline.py 的 assemble_text）。
"""
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as _np
import onnxruntime as _ort

log = logging.getLogger("openocr-openai")
if not log.handlers:            # uvicorn 只配置自己的 logger，我们的 INFO 日志默认会被丢掉
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [openocr-openai] %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("OPENOCR_LOG_LEVEL", "INFO").upper())

os.environ.setdefault("OMP_NUM_THREADS", "4")

# ---- 推理设备与 CPU 线程上限 -------------------------------------------------
# openocr 建 ONNX 会话时不设 intra_op_num_threads，ORT 默认按机器全部核心开线程
# （24 核 = 每个会话 24 线程）。这台机器是共享的（llama.cpp、strata 等同机跑），
# CPU 一忙，ORT 的自旋等待线程就互相抢核：实测 loadavg=10 时
#   单行小图 0.41s -> 1.10s，表格 4.5s -> 11.8s，整页 5.1s -> 11.4s（2.3~2.5 倍）。
# 限到 8 线程后同样的负载下延迟稳定；空闲时与默认一致；纯线程数，精度不变。
# torch 侧用同一个值（torch.set_num_threads），两个后端对齐，不会互相超订。
# 设 0 = 用各自默认（不建议在共享机上用）。
GPU = os.environ.get("OPENOCR_USE_GPU", "false").lower() in ("1", "true", "yes", "auto")
CPU_THREADS = int(os.environ.get("OPENOCR_CPU_THREADS", "0"))
if CPU_THREADS == 0 and not GPU:
    CPU_THREADS = min(8, os.cpu_count() or 8)

# openocr 的推理日志里有 emoji（🔧 ✅），Windows 下 stdout 被重定向到文件/服务时默认 GBK，
# 一个 emoji 就能让进程 UnicodeEncodeError 直接退出。统一按 UTF-8 输出，编码不了的字符替换掉。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_orig_session = _ort.InferenceSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 对外只暴露一个模型 id。旧 id 作为 alias 收进来：new-api 上已配好的渠道不用改，
# 但 /v1/models 只列规范 id，避免网关下拉列表里出现三个一样的引擎。
MODEL = "scriptmoe"
MODELS = [MODEL]
MODEL_ALIASES = {"opendoc-0.1b": MODEL, "unirec-0.1b": MODEL, "scriptmoe-0.1b": MODEL}

# ---- ScriptMoE 引擎配置 ------------------------------------------------------
# 权重不进 git（191MB safetensors），从上游仓库取：
#   https://github.com/YesianRohn/ScriptMoE  ->  E2EOCR/model.safetensors + E2EOCR/assets/dict.txt
# 放到 models/scriptmoe/ 下即可，或用下面两个环境变量指到别处。
SCRIPTMOE_WEIGHTS = os.environ.get(
    "SCRIPTMOE_WEIGHTS", os.path.join(BASE_DIR, "models", "scriptmoe", "model.safetensors"))
SCRIPTMOE_DICT = os.environ.get(
    "SCRIPTMOE_DICT", os.path.join(BASE_DIR, "models", "scriptmoe", "dict.txt"))
# 检测模型（repvit_db ONNX）：留空 = 让 openocr 自己下载并缓存到 ~/.cache/openocr。
# 容器/离线环境提前把 onnx 放进挂载目录，再用这个变量指过去。
SCRIPTMOE_DET_MODEL = os.environ.get("SCRIPTMOE_DET_MODEL", "")
# 行宽高比上限：识别张量宽度 = 32 x ratio，encoder 第三段是全局注意力，
# 单图显存与耗时都随它平方增长。20 = 上游默认，证照/表单这类窄条文字够用。
SCRIPTMOE_MAX_RATIO = int(os.environ.get("SCRIPTMOE_MAX_RATIO", "20"))
# 单行最多识别多少个字符（decoder 自回归步数上限）。
SCRIPTMOE_MAX_TEXT_LENGTH = int(os.environ.get("SCRIPTMOE_MAX_TEXT_LENGTH", "100"))
# 低于这个识别置信度的行直接丢（上游默认 0.5）。
SCRIPTMOE_DROP_SCORE = float(os.environ.get("SCRIPTMOE_DROP_SCORE", "0.5"))
# 一次拼多少个裁条进 batch（同宽高比才拼得进去）。显存 = 权重 + batch x 单图激活，
# 1050 Ti 4GB 实测 batch 8 x ratio 20 ≈ 1.4GiB 激活，见 README「性能与显存」。
SCRIPTMOE_REC_BATCH = int(os.environ.get("SCRIPTMOE_REC_BATCH", "8"))
# 检测器规格：mobile = repvit_db 轻量检测（12MB ONNX）。
SCRIPTMOE_DET_MODE = os.environ.get("SCRIPTMOE_DET_MODE", "mobile")
# 输出层兜底：单行结尾被同一个子串刷屏时截断（ScriptMoE 是定长 100 步解码，
# 一行烧穿的表现就是 '////'、'159595959…' 这种）。阈值 = 同一子串连续重复次数。
TRIM_MIN_TIMES = int(os.environ.get("OPENOCR_TRIM_MIN_TIMES", "8"))


def _session(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
    """ORT 会话工厂：GPU 压显存 arena，CPU 限线程数（见上面 CPU_THREADS 注释）。

    注意 ORT 1.22 已移除 enable_mem_arena / device_mem_limit，写进 provider_options
    会让 CUDA EP 注册失败并静默回退 CPU。
    """
    if providers and "CUDAExecutionProvider" in providers:
        provider_options = [
            {"arena_extend_strategy": "kSameAsRequested",
             "cudnn_conv_use_max_workspace": "0"} if p == "CUDAExecutionProvider" else {}
            for p in providers
        ]
        if sess_options is None:
            sess_options = _ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
    if CPU_THREADS > 0 and not (providers and "CUDAExecutionProvider" in providers):
        if sess_options is None:
            sess_options = _ort.SessionOptions()
        sess_options.intra_op_num_threads = CPU_THREADS
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
    return _orig_session(path_or_bytes, sess_options, providers=providers,
                         provider_options=provider_options, **kw)


_ort.InferenceSession = _session


def _trim_repeats(text: str) -> str:
    """单行文本结尾的重复刷屏截断（引擎输出是逐行文字，没有表格骨架要保护）。"""
    n = len(text)
    if n < 24:
        return text
    for L in range(1, 25):
        unit = text[-L:]
        if n < L * TRIM_MIN_TIMES:
            continue
        reps = 1
        i = n - L
        while i - L >= 0 and text[i - L:i] == unit:
            reps += 1
            i -= L
        if reps >= TRIM_MIN_TIMES:
            return text[:i + L].rstrip() + "…"
    return text


# 前置缩放：默认关闭（0 = 原图直接进模型）。识别裁条会被统一压到 32 高的固定分辨率，
# 检测器自己也按固定边长缩放，所以缩放对精度没有好处，只当解码炸弹的保险丝用：
# 一张 20MB 的 JPEG 解出来可能是 10000x10000 = 300MB 内存，这时把它设成 4096 之类。
# 精度侧的实测结论（462 张营业执照）：原图 > 2048px > 1024px，能原图就原图。
MAX_PIXEL_SIDE = int(os.environ.get("OPENOCR_MAX_PIXEL_SIDE", "0"))
# 单图体积上限（base64 解码后的原始字节），超过则重编码 JPEG。
MAX_IMAGE_BYTES = int(os.environ.get("OPENOCR_MAX_IMAGE_KB", "5120")) * 1024
# 远端图片：下载超时与体积上限（无上限时一张 500MB 的图能把内存打满）
URL_TIMEOUT = int(os.environ.get("OPENOCR_URL_TIMEOUT", "30"))
MAX_URL_BYTES = int(os.environ.get("OPENOCR_MAX_URL_MB", "20")) * 1024 * 1024
# 请求体上限（base64 膨胀 4/3，64MB 请求体约等于 48MB 图片）
MAX_BODY_BYTES = int(os.environ.get("OPENOCR_MAX_BODY_MB", "64")) * 1024 * 1024


def _shrink(path: str) -> str:
    """超限图片等比压缩为 JPEG（<=MAX_IMAGE_BYTES），返回新路径并删除原文件。"""
    from PIL import Image, ImageOps
    try:
        im = Image.open(path)
        im.load()
    except Exception:
        return path
    orig_size = im.size
    # EXIF 方向：手机/微信拍的证照绝大多数带 Orientation 标签（6=顺时针 90°），
    # 而检测/识别侧用 cv2.imdecode 读图，**不看 EXIF**，等于把图旋转 90° 喂给模型，
    # 整页识别质量直接崩。这里统一按 EXIF 转正（无标签时是空操作）。
    # 需要转正时 (w, h) != orig_size 一定成立，所以一定会走到下面的重编码分支落盘，
    # 不存在「转正了但没存盘」的中间态。
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    w, h = im.size
    need = (os.path.getsize(path) > MAX_IMAGE_BYTES
             or (MAX_PIXEL_SIDE > 0 and max(w, h) > MAX_PIXEL_SIDE)
             or (w, h) != orig_size)
    if not need:
        return path
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    side = max(im.size)
    if MAX_PIXEL_SIDE > 0 and side > MAX_PIXEL_SIDE:
        r = MAX_PIXEL_SIDE / float(side)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    out = path + ".c.jpg"
    q = 92
    while True:
        im.save(out, "JPEG", quality=q, optimize=True)
        if os.path.getsize(out) <= MAX_IMAGE_BYTES or q <= 55:
            break
        q -= 8
    try:
        os.remove(path)
    except OSError:
        pass
    return out


_engine = None
_engine_lock = threading.Lock()   # 防止并发首次请求把模型加载两遍（内存翻倍）
_lock = threading.Lock()          # 单实例串行推理，防止显存/内存爆掉


def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            from ocr_engine import ScriptMoEOcr
            _engine = ScriptMoEOcr(
                weights_path=SCRIPTMOE_WEIGHTS,
                char_dict_path=SCRIPTMOE_DICT,
                use_gpu="true" if GPU else "false",
                det_mode=SCRIPTMOE_DET_MODE,
                det_model_path=SCRIPTMOE_DET_MODEL or None,
                drop_score=SCRIPTMOE_DROP_SCORE,
                rec_batch_num=SCRIPTMOE_REC_BATCH,
                max_text_length=SCRIPTMOE_MAX_TEXT_LENGTH,
                max_ratio=SCRIPTMOE_MAX_RATIO,
                threads=CPU_THREADS,
            )
            log.info("ScriptMoE 已加载: device=%s det=%s weights=%s (%.1fs)",
                     _engine.device, SCRIPTMOE_DET_MODE, SCRIPTMOE_WEIGHTS,
                     _engine.load_seconds)
    return _engine


def _run_scriptmoe(engine, img_path: str):
    """单图推理 -> (纯文本, 耗时明细)。逐行做重复截断后按阅读顺序重新组装。"""
    from ocr_engine import assemble_text
    text, lines, timing = engine.infer_file(img_path)
    if lines:
        text = assemble_text([dict(l, text=_trim_repeats(l["text"])) for l in lines])
    return text, timing


def _warmup():
    """启动时加载模型 + 跑一张极小图。

    只加载模型是不够的：ORT 的图优化产物、内存 arena、torch 首次算子分发都是首次推理
    才建，不预热的话第一个真实请求要额外付 1~3s（GPU 更多），单图场景下尾巴很显眼。
    """
    from PIL import Image
    eng = get_engine()
    t0 = time.time()
    fd, probe = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(_np.full((48, 160, 3), 255, _np.uint8)).save(probe)
        _run_scriptmoe(eng, probe)
        log.info("预热完成: %.1fs", time.time() - t0)
    except Exception as exc:      # 预热失败不能拖垮启动
        log.warning("预热失败（不影响启动）: %s", exc)
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warmup()          # 启动时加载模型，就绪后 /health 才返回 ok
    yield


app = FastAPI(title="ScriptMoE OpenAI-compatible API", version="2.0", lifespan=lifespan)


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    """请求体上限：base64 图片直接进 JSON body，不设上限时一个大请求就能把内存打满。"""
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return _err(413, "请求体 " + cl + " 字节，超过上限 "
                    + str(MAX_BODY_BYTES // (1024 * 1024)) + "MB")
    return await call_next(request)


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None


class ChatCompletionRequest(BaseModel):
    # max_tokens 是 OpenAI 协议的习惯字段：接受但不参与推理控制。ScriptMoE 是
    # 「逐行定长解码」：单行上限 SCRIPTMOE_MAX_TEXT_LENGTH 个字符，行数由检测框决定，
    # 没有全局 token 预算可以截断。
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 2048
    stream: bool = False


def _err(status: int, message: str, etype: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": etype}})


def _fetch_url(url: str, suffix: str) -> str:
    """下载远端图片：带超时与体积上限。urlretrieve 无超时，URL 卡住会永久占住 worker 线程。"""
    req = urllib.request.Request(url, headers={"User-Agent": "openocr-openai/2.0"})
    p = os.path.join(tempfile.gettempdir(), "oocr_" + uuid.uuid4().hex + suffix)
    total = 0
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT) as resp, open(p, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_URL_BYTES:
                raise ValueError("远端图片超过 " + str(MAX_URL_BYTES // (1024 * 1024)) + "MB 上限")
            f.write(chunk)
    if total == 0:
        raise ValueError("远端图片为空")
    return p


def _extract_images(messages: list[ChatMessage]) -> list[str]:
    """从 messages 中提取所有图片(data URI 或 http url)，返回临时文件路径列表。"""
    paths: list[str] = []
    for m in messages:
        if not isinstance(m.content, list):
            continue
        for part in m.content:
            if part.type != "image_url" or not part.image_url:
                continue
            url = part.image_url.get("url", "")
            if url.startswith("data:"):
                header, b64 = url.split(",", 1)
                ext = header.split(";")[0].split("/")[-1].split("+")[0]
                suffix = {"jpeg": "jpg"}.get(ext, ext) or "png"
                p = os.path.join(tempfile.gettempdir(), "oocr_" + uuid.uuid4().hex + "." + suffix)
                with open(p, "wb") as f:
                    f.write(base64.b64decode(b64))
                paths.append(p)
            elif url.startswith(("http://", "https://")):
                suffix = os.path.splitext(url.split("?")[0])[1] or ".png"
                paths.append(_fetch_url(url, suffix))
            else:
                raise ValueError("不支持的 image_url: " + url[:80])
    return paths


def _notice(model: str) -> str:
    """无图片请求（网关连通性测试）的说明文本。"""
    return (
        "ScriptMoE 服务在线，模型 " + model + " 可用。\n\n"
        "这是纯 OCR 模型，只接受图片输入：请在 messages 的 content 数组里放 image_url"
        "（base64 data URI 或 http(s) 链接），支持单请求多图。\n"
        "可用模型：" + ", ".join(MODELS) + "（文本检测 + ScriptMoE 多语种识别，"
        "输出按阅读顺序组装的纯文本，单请求多图）。\n"
        "本条是无图片请求的连通性应答，未执行推理。"
    )


def _sse(chunk: dict) -> str:
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def _answer(model: str, content: str, stream: bool):
    """非流式返回 chat.completion，流式返回 chat.completion.chunk 序列。"""
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    if not stream:
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def gen():
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        yield _sse(dict(base, choices=[{"index": 0,
                                        "delta": {"role": "assistant", "content": ""},
                                        "finish_reason": None}]))
        step = 512
        for i in range(0, max(1, len(content)), step):
            yield _sse(dict(base, choices=[{"index": 0,
                                            "delta": {"content": content[i:i + step]},
                                            "finish_reason": None}]))
        yield _sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/health")
def health():
    return {"status": "ok", "models": MODELS, "loaded": [MODEL] if _engine else [],
            "gpu": GPU, "device": _engine.device if _engine else None,
            "det_mode": SCRIPTMOE_DET_MODE,
            "max_ratio": SCRIPTMOE_MAX_RATIO,
            "max_text_length": SCRIPTMOE_MAX_TEXT_LENGTH,
            "rec_batch": SCRIPTMOE_REC_BATCH,
            "drop_score": SCRIPTMOE_DROP_SCORE,
            "cpu_threads": CPU_THREADS,
            "weights": SCRIPTMOE_WEIGHTS,
            "char_dict": SCRIPTMOE_DICT,
            "det_model": SCRIPTMOE_DET_MODEL or "auto-download",
            "max_pixel_side": MAX_PIXEL_SIDE,
            "max_image_kb": MAX_IMAGE_BYTES // 1024,
            "trim_min_times": TRIM_MIN_TIMES,
            "url_timeout_s": URL_TIMEOUT, "max_url_mb": MAX_URL_BYTES // (1024 * 1024),
            "max_body_mb": MAX_BODY_BYTES // (1024 * 1024)}


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": m, "object": "model", "created": int(time.time()),
                      "owned_by": "scriptmoe"} for m in MODELS]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model = MODEL_ALIASES.get(req.model, req.model)
    if model not in MODELS:
        return _err(404, "模型不存在: " + str(req.model) + "，可用: " + str(MODELS),
                    "model_not_found")
    try:
        images = _extract_images(req.messages)
    except Exception as e:
        return _err(400, "图片解析失败: " + str(e))

    # 无图片：返回 200 说明文本。网关（new-api 等）连通性测试只发纯文本，
    # 这里回 400 会让渠道测试判定失败，所以必须 200。
    if not images:
        return _answer(model, _notice(model), req.stream)

    images = [_shrink(p) for p in images]
    engine = get_engine()
    contents = []
    t0 = time.time()
    det_ms = rec_ms = 0.0
    n_boxes = n_kept = 0
    try:
        with _lock:
            for i, p in enumerate(images):
                text, timing = _run_scriptmoe(engine, p)
                det_ms += timing["det_ms"]
                rec_ms += timing["rec_ms"]
                n_boxes += timing["num_boxes"]
                n_kept += timing["num_kept"]
                name = os.path.basename(p)
                header = "### 图片 " + str(i + 1) + "/" + str(len(images)) + ": " + name + "\n\n"
                contents.append(header + text.strip() if len(images) > 1 else text.strip())
    except Exception as e:
        log.exception("推理失败")
        return _err(500, "推理失败: " + str(e), "server_error")
    finally:
        for p in images:
            try:
                os.remove(p)
            except OSError:
                pass

    log.info("%s | %d 图 | 框 %d 留 %d | det %.0fms rec %.0fms | 总计 %.1fs | %d 字",
             model, len(images), n_boxes, n_kept, det_ms, rec_ms,
             time.time() - t0, sum(len(c) for c in contents))
    return _answer(model, "\n\n---\n\n".join(contents), req.stream)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8089")))
