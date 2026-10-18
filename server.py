"""OpenAI-compatible API server for OpenOCR (OpenDoc / UniRec), images only, multi-image supported.

设计要点（接入 new-api 等 OpenAI 兼容网关）：
- 纯文本请求（网关连通性测试）返回 200 + 说明文本，不再返回 400，否则渠道测试过不了。
- 支持 stream=true，按 chat.completion.chunk 吐 SSE，结尾 data: [DONE]。
- 图片走 content 数组里的 image_url（base64 data URI 或 http(s)），单请求多图。
"""
import base64
import json
import logging
import os
import re
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

# openocr 的推理日志里有 emoji（🔧 ✅），Windows 下 stdout 被重定向到文件/服务时默认 GBK，
# 一个 emoji 就能让进程 UnicodeEncodeError 直接退出。统一按 UTF-8 输出，编码不了的字符替换掉。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_orig_session = _ort.InferenceSession

MODELS = ["opendoc-0.1b", "unirec-0.1b"]
MAX_TOKENS_CAP = 2048

# 压缩 ONNX Runtime CUDA 显存：4 个会话（opendoc 布局+识别 / unirec 编码+解码）各有独立 arena，
# arena 只增不还，瞬时峰值会永久抬高上限，默认实测稳态 3.7~4.2GB。
# 下面关掉 memory pattern 预分配、不让 cuDNN 申请最大卷积 workspace，arena 改为按需扩展。
# 注意 ORT 1.22 已移除 enable_mem_arena / device_mem_limit，写进 provider_options 会让 CUDA EP
# 注册失败并静默回退 CPU。


def _session(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
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
    return _orig_session(path_or_bytes, sess_options, providers=providers,
                         provider_options=provider_options, **kw)


_ort.InferenceSession = _session

# ---------------------------------------------------------------------------
# UniRec 解码层补丁：重复循环早停 + 去掉每步的张量拷贝
#
# 库原版解码循环（openocr/tools/infer_unirec_onnx.py 的 _infer_single_image）是纯贪心 argmax，
# 出口只有 EOS 一个判断，没有任何重复检测；上游后处理也只有 `_{4,}`→`___`、`\.{4,}`→`...` 两条正则，
# 治不了 token 级循环。模型一旦陷入循环（空白块、密集小字、超宽长条最容易触发），
# 会一路跑满 max_length 步（默认 2048），单张图白烧几十秒，还全程占着服务端的串行锁，
# 整个服务被一个请求拖死。
#
# 这里在运行时给 UniRecONNX.decode_step 打补丁（不改 site-packages，升级 openocr 后签名不匹配就自动跳过）：
#   1) 尾部 n-gram 连续重复 k 次 → 把该步 logits 的 argmax 改成 EOS，让库的循环自然 break；
#   2) 原版每步都对 cross_k/cross_v（整页 encoder 的 KV）和每层 past_key/past_value 无条件
#      .astype(np.float32)，dtype 本来就是 float32 也照拷。整页 1024px 时 cross KV 有几十 MB，
#      一次生成几百步就是 GB 级 memcpy，长文本下这块能吃掉解码时间的两三成。
# ---------------------------------------------------------------------------
REPEAT_GUARD = os.environ.get("OPENOCR_REPEAT_GUARD", "true").lower() in ("1", "true", "yes")
# 判定方式：找尾部的最小重复周期 p（1..REPEAT_NGRAM），要求「同一周期连续重复 >= REPEAT_MIN_REPS 次，
# 且累计 token 数 >= 阈值」才算循环。用周期而不是固定 n-gram，才能抓到 <td>金额</td> 这种 3 token 周期的小循环。
REPEAT_NGRAM = int(os.environ.get("OPENOCR_REPEAT_NGRAM", "8"))          # 检测的最大周期
REPEAT_MIN_REPS = int(os.environ.get("OPENOCR_REPEAT_MIN_REPS", "3"))    # 至少完整重复几遍
REPEAT_MIN_TOKENS = int(os.environ.get("OPENOCR_REPEAT_MIN_TOKENS", "32"))        # 正文：连续重复这么多 token 判定循环
# 表格专用：<td></td>、|---|、|  | 这类纯结构 token 的重复是表格骨架本身，阈值放宽 3 倍
# （96 token ≈ 48 个连续空单元格），否则空单元格多的表会被误截。
REPEAT_STRUCT_MIN_TOKENS = int(os.environ.get("OPENOCR_REPEAT_STRUCT_MIN_TOKENS", "96"))
# 输出层兜底（解码层早停之外的补漏），同样区分正文/表格骨架
TRIM_MIN_TIMES = int(os.environ.get("OPENOCR_TRIM_MIN_TIMES", "8"))
TRIM_STRUCT_MIN_TIMES = int(os.environ.get("OPENOCR_TRIM_STRUCT_MIN_TIMES", "24"))

# 去掉 HTML 标签后只剩分隔符/空白/标点的片段 = 纯结构（表格骨架），不含任何文字
_TAG_RE = re.compile(r"<[^>]*>")
_CONTENT_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _is_structural(text: str) -> bool:
    """这段文本是不是纯表格骨架（<td></td>、|---|、|  | 之类，没有任何实际文字/数字）。"""
    stripped = _TAG_RE.sub("", text)
    if not stripped.strip():
        return True
    return not _CONTENT_RE.search(stripped)


_repeat_state = threading.local()   # 每个 worker 线程一次只跑一个生成（doc 任务多线程共享同一 session）


def _count_tail_reps(seq, unit_len: int) -> int:
    """尾部 unit_len 长度的片段连续重复了几次。"""
    L = len(seq)
    if unit_len <= 0 or L < unit_len:
        return 0
    tail = seq[-unit_len:]
    reps = 1
    while L - unit_len * (reps + 1) >= 0 and seq[L - unit_len * (reps + 1):L - unit_len * reps] == tail:
        reps += 1
    return reps


def _tail_period(seq):
    """尾部最小重复周期及其重复次数：返回 (period, reps)，无重复时 (1, 1)。"""
    for p in range(1, min(REPEAT_NGRAM, len(seq)) + 1):
        reps = _count_tail_reps(seq, p)
        if reps >= REPEAT_MIN_REPS:
            return p, reps
    return 1, 1


def _is_repeating(seq, texts=None) -> bool:
    """seq 尾部是否已进入重复循环。texts 是对应 token 的文本，用来区分表格骨架与正文。

    表格输出天然高度周期化（每行结构一样），所以只看「重复得有多长」会误伤正常表格：
    判定单位是连续重复的 token 总数，且纯结构（没有任何文字/数字）的重复阈值放宽 3 倍。
    """
    p, reps = _tail_period(seq)
    if reps < REPEAT_MIN_REPS:
        return False
    unit_txt = "".join(texts[-p:]) if texts else ""
    structural = bool(texts) and _is_structural(unit_txt)
    need = REPEAT_STRUCT_MIN_TOKENS if structural else REPEAT_MIN_TOKENS
    return p * reps >= need


def _trim_repeats(text: str) -> str:
    """输出层兜底：结尾被同一个子串刷屏时截断（解码层早停之外的补漏）。

    纯结构重复（表格骨架）阈值放宽到 TRIM_STRUCT_MIN_TIMES，避免误截空单元格多的表。
    """
    n = len(text)
    if n < 24:
        return text
    for L in range(1, 25):
        unit = text[-L:]
        need = TRIM_STRUCT_MIN_TIMES if _is_structural(unit) else TRIM_MIN_TIMES
        if n < L * need:
            continue
        reps = 1
        i = n - L
        while i - L >= 0 and text[i - L:i] == unit:
            reps += 1
            i -= L
        if reps >= need:
            return text[:i + L].rstrip() + "…"
    return text


def _patch_unirec(cls) -> None:
    """给 UniRecONNX 类打补丁（doc 任务的 vlm_recognizer 也是这个类，一处覆盖两个引擎）。"""
    if not hasattr(cls, "decode_step") or getattr(cls.decode_step, "_ocr_patched", False):
        return
    import inspect
    params = list(inspect.signature(cls.decode_step).parameters)
    if params[:6] != ["self", "input_id", "past_length", "cross_k", "cross_v", "past_key_values"]:
        log.warning("UniRecONNX.decode_step 签名已变，跳过解码补丁（重复循环防护与拷贝优化不生效）")
        return

    def decode_step(self, input_id, past_length, cross_k, cross_v, past_key_values, padding_idx=1):
        hist = getattr(_repeat_state, "hist", None)
        if hist is None or past_length == 0:
            hist = []
            _repeat_state.hist = hist
            _repeat_state.texts = []
            _repeat_state.steps = 0
            _repeat_state.stopped = False
        tid = int(input_id)
        hist.append(tid)
        _repeat_state.steps = past_length + 1
        id2tok = getattr(self.tokenizer, "id_to_token", None)
        _repeat_state.texts.append(id2tok.get(tid, "") if isinstance(id2tok, dict) else "")

        f32 = _np.float32
        ck = cross_k if cross_k.dtype == f32 else cross_k.astype(f32)
        cv = cross_v if cross_v.dtype == f32 else cross_v.astype(f32)
        cache = [(k, v) if k.dtype == f32 and v.dtype == f32
                 else (k.astype(f32), v.astype(f32)) for k, v in past_key_values]

        logits, present = self._ocr_orig_decode_step(input_id, past_length, ck, cv, cache,
                                                     padding_idx=padding_idx)

        if REPEAT_GUARD and len(hist) >= REPEAT_MIN_TOKENS \
                and _is_repeating(hist, _repeat_state.texts):
            eos = int(getattr(self.tokenizer, "eos_token_id", 2))
            row = logits[0, -1]
            if 0 <= eos < row.shape[0]:
                row[:] = -1e30
                row[eos] = 1.0
                _repeat_state.stopped = True
                log.warning("检测到重复输出，已在第 %d 步强制结束（连续重复 %d token，周期 %d，正文阈值 %d / 表格骨架阈值 %d）",
                            past_length + 1, len(hist), _tail_period(hist)[0],
                            REPEAT_MIN_TOKENS, REPEAT_STRUCT_MIN_TOKENS)
        return logits, present

    decode_step._ocr_patched = True
    cls._ocr_orig_decode_step = cls.decode_step
    cls.decode_step = decode_step
    log.info("UniRec 解码补丁已生效：重复循环早停 周期<=%d 且连续重复>=%d token（表格骨架 %d token）",
             REPEAT_NGRAM, REPEAT_MIN_TOKENS, REPEAT_STRUCT_MIN_TOKENS)


# 前置缩放压显存（不改 openocr 任何配置）。
# UniRec 的 SimpleImageProcessor 只做"超过 max_side(960x1408) 才缩小"，输入小于该尺寸时原样使用。
# encoder 自注意力显存随 token 数增长，所以在进模型前把最长边压到 MAX_PIXEL_SIDE，模型侧参数一个都不动。
MAX_PIXEL_SIDE = int(os.environ.get("OPENOCR_MAX_PIXEL_SIDE", "1024"))
# 单图体积上限（base64 解码后的原始字节），超过则重编码 JPEG。
MAX_IMAGE_BYTES = int(os.environ.get("OPENOCR_MAX_IMAGE_KB", "5120")) * 1024
# 远端图片：下载超时与体积上限（无上限时一张 500MB 的图能把内存打满）
URL_TIMEOUT = int(os.environ.get("OPENOCR_URL_TIMEOUT", "30"))
MAX_URL_BYTES = int(os.environ.get("OPENOCR_MAX_URL_MB", "20")) * 1024 * 1024
# 请求体上限（base64 膨胀 4/3，64MB 请求体约等于 48MB 图片）
MAX_BODY_BYTES = int(os.environ.get("OPENOCR_MAX_BODY_MB", "64")) * 1024 * 1024


def _shrink(path: str) -> str:
    """超限图片等比压缩为 JPEG（<=MAX_IMAGE_BYTES），返回新路径并删除原文件。"""
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
    except Exception:
        return path
    w, h = im.size
    need = os.path.getsize(path) > MAX_IMAGE_BYTES or max(w, h) > MAX_PIXEL_SIDE
    if not need:
        return path
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    side = max(im.size)
    if side > MAX_PIXEL_SIDE:
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


_engines: dict = {}
_engine_lock = threading.Lock()   # 防止并发首次请求把同一个模型加载两遍（内存翻倍）
GPU = os.environ.get("OPENOCR_USE_GPU", "false").lower() in ("1", "true", "yes", "auto")
_lock = threading.Lock()          # 单实例串行推理，防止显存/内存爆掉
# doc 任务内部按版面块并行识别（共享同一个 ONNX 会话，ORT 会话线程安全）。
# CPU 下并行是真提速（session.run 会释放 GIL）；GPU 下并行几乎不涨吞吐，
# 只会抬高解码峰值显存，所以默认 CPU=4、GPU=1。
MAX_PARALLEL_BLOCKS = int(os.environ.get(
    "OPENOCR_MAX_PARALLEL_BLOCKS", "1" if GPU else "4"))


def get_engine(model: str):
    if model in _engines:
        return _engines[model]
    with _engine_lock:
        if model in _engines:
            return _engines[model]
        from openocr import OpenOCR
        if model == "opendoc-0.1b":
            eng = OpenOCR(task="doc", use_layout_detection=True, use_gpu=GPU,
                          max_parallel_blocks=MAX_PARALLEL_BLOCKS)
        elif model == "unirec-0.1b":
            eng = OpenOCR(task="unirec", use_gpu=GPU)
        else:
            raise KeyError(model)
        # doc 引擎的识别器是引擎内部的 vlm_recognizer，unirec 引擎本身就是 UniRecONNX
        rec = getattr(getattr(eng, "model", None), "vlm_recognizer", None) \
            if model == "opendoc-0.1b" else getattr(eng, "model", None)
        if rec is not None:
            _patch_unirec(type(rec))
        _engines[model] = eng
        return eng


def _warmup():
    for m in MODELS:
        get_engine(m)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warmup()          # 启动时加载模型，就绪后 /health 才返回 ok
    yield


app = FastAPI(title="OpenOCR OpenAI-compatible API", version="1.2", lifespan=lifespan)


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    """请求体上限：base64 图片直接进 JSON body，不设上限时一个大请求就能把内存打满。"""
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return _err(413, f"请求体 {cl} 字节，超过上限 {MAX_BODY_BYTES // (1024 * 1024)}MB")
    return await call_next(request)


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 2048
    stream: bool = False


def _err(status: int, message: str, etype: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": etype}})


def _fetch_url(url: str, suffix: str) -> str:
    """下载远端图片：带超时与体积上限。原版 urlretrieve 无超时，URL 卡住会永久占住 worker 线程。"""
    req = urllib.request.Request(url, headers={"User-Agent": "openocr-openai/1.2"})
    p = os.path.join(tempfile.gettempdir(), f"oocr_{uuid.uuid4().hex}{suffix}")
    total = 0
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT) as resp, open(p, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_URL_BYTES:
                raise ValueError(f"远端图片超过 {MAX_URL_BYTES // (1024 * 1024)}MB 上限")
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
                p = os.path.join(tempfile.gettempdir(), f"oocr_{uuid.uuid4().hex}.{suffix}")
                with open(p, "wb") as f:
                    f.write(base64.b64decode(b64))
                paths.append(p)
            elif url.startswith(("http://", "https://")):
                suffix = os.path.splitext(url.split("?")[0])[1] or ".png"
                paths.append(_fetch_url(url, suffix))
            else:
                raise ValueError(f"不支持的 image_url: {url[:80]}")
    return paths


def _run_opendoc(engine, img_path: str) -> str:
    result = engine(image_path=img_path)
    pages = result if isinstance(result, list) else [result]
    outs = []
    for page in pages:
        with tempfile.TemporaryDirectory() as td:
            engine.save_to_markdown(page, td)
            for root, _dirs, files in os.walk(td):
                for fn in files:
                    if fn.endswith(".md"):
                        with open(os.path.join(root, fn), encoding="utf-8") as f:
                            outs.append(f.read())
    return "\n\n".join(outs)


def _run_unirec(engine, img_path: str, max_tokens: int) -> str:
    _repeat_state.stopped = False
    _repeat_state.steps = 0
    text, _ = engine(image_path=img_path, max_length=max_tokens)
    text = _trim_repeats(text)
    if getattr(_repeat_state, "stopped", False):
        log.warning("unirec 输出被重复防护截断: %s", os.path.basename(img_path))
    # 跑满 token 预算说明是被长度切断的，不是模型自然收尾 —— 表格/长文档最常见，
    # 静默截断会让调用方拿到半张表，所以显式标记出来。
    if getattr(_repeat_state, "steps", 0) >= max_tokens - 1:
        log.warning("unirec 输出达到 max_tokens=%d 上限，已标记截断: %s", max_tokens, os.path.basename(img_path))
        text += (f"\n\n> ⚠️ 输出达到 max_tokens={max_tokens} 上限，内容被截断。"
                 f"表格/长文档请把大图按行块切分后分多次请求（服务端上限 {MAX_TOKENS_CAP}）。")
    return text


def _notice(model: str) -> str:
    """无图片请求（网关连通性测试）的说明文本。"""
    return (
        f"OpenOCR 服务在线，模型 `{model}` 可用。\n\n"
        "这是纯 OCR 模型，只接受图片输入：请在 messages 的 content 数组里放 image_url"
        "（base64 data URI 或 http(s) 链接），支持单请求多图。\n"
        f"可用模型：{', '.join(MODELS)}（opendoc-0.1b=整页文档转 Markdown，unirec-0.1b=单行/单块文字识别）。\n"
        "本条是无图片请求的连通性应答，未执行推理。"
    )


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _answer(model: str, content: str, stream: bool):
    """统一出口：非流式返回 chat.completion，流式返回 chat.completion.chunk 序列。"""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
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
    return {"status": "ok", "models": MODELS, "loaded": sorted(_engines),
            "gpu": GPU, "max_pixel_side": MAX_PIXEL_SIDE,
            "max_image_kb": MAX_IMAGE_BYTES // 1024,
            "max_parallel_blocks": MAX_PARALLEL_BLOCKS,
            "repeat_guard": REPEAT_GUARD,
            "repeat_min_tokens": REPEAT_MIN_TOKENS,
            "repeat_struct_min_tokens": REPEAT_STRUCT_MIN_TOKENS,
            "url_timeout_s": URL_TIMEOUT, "max_url_mb": MAX_URL_BYTES // (1024 * 1024),
            "max_body_mb": MAX_BODY_BYTES // (1024 * 1024)}


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": m, "object": "model", "created": int(time.time()),
                      "owned_by": "openocr"} for m in MODELS]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.model not in MODELS:
        return _err(404, f"模型不存在: {req.model}，可用: {MODELS}", "model_not_found")
    max_tokens = req.max_tokens or 2048
    max_tokens = max(1, min(int(max_tokens), MAX_TOKENS_CAP))
    try:
        images = _extract_images(req.messages)
    except Exception as e:
        return _err(400, f"图片解析失败: {e}")

    # 无图片：返回 200 说明文本。网关（new-api 等）连通性测试只发纯文本，
    # 这里回 400 会让渠道测试判定失败，所以必须 200。
    if not images:
        return _answer(req.model, _notice(req.model), req.stream)

    images = [_shrink(p) for p in images]
    engine = get_engine(req.model)
    contents = []
    t0 = time.time()
    try:
        with _lock:
            for i, p in enumerate(images):
                if req.model == "unirec-0.1b":
                    text = _run_unirec(engine, p, max_tokens)
                else:
                    text = _run_opendoc(engine, p)
                name = os.path.basename(p)
                header = f"### 图片 {i + 1}/{len(images)}: {name}\n\n" if len(images) > 1 else ""
                contents.append(header + text.strip())
    except Exception as e:
        log.exception("推理失败")
        return _err(500, f"推理失败: {e}", "server_error")
    finally:
        for p in images:
            try:
                os.remove(p)
            except OSError:
                pass

    log.info("%s | %d 图 | %.1fs | %d 字", req.model, len(images), time.time() - t0,
             sum(len(c) for c in contents))
    return _answer(req.model, "\n\n---\n\n".join(contents), req.stream)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8089")))