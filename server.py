"""OpenAI-compatible API server for OpenOCR (OpenDoc / UniRec), images only, multi-image supported.

设计要点（接入 new-api 等 OpenAI 兼容网关）：
- 纯文本请求（网关连通性测试）返回 200 + 说明文本，不再返回 400，否则渠道测试过不了。
- 支持 stream=true，按 chat.completion.chunk 吐 SSE，结尾 data: [DONE]。
- 图片走 content 数组里的 image_url（base64 data URI 或 http(s)），单请求多图。
"""
import base64
import json
import os
import tempfile
import time
import uuid
import threading

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import onnxruntime as _ort

os.environ.setdefault("OMP_NUM_THREADS", "4")

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

# 前置缩放压显存（不改 openocr 任何配置）。
# UniRec 的 SimpleImageProcessor 只做"超过 max_side(960x1408) 才缩小"，输入小于该尺寸时原样使用。
# encoder 自注意力显存随 token 数增长，所以在进模型前把最长边压到 MAX_PIXEL_SIDE，模型侧参数一个都不动。
MAX_PIXEL_SIDE = int(os.environ.get("OPENOCR_MAX_PIXEL_SIDE", "1024"))
# 单图体积上限（base64 解码后的原始字节），超过则重编码 JPEG。
MAX_IMAGE_BYTES = int(os.environ.get("OPENOCR_MAX_IMAGE_KB", "5120")) * 1024


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
GPU = os.environ.get("OPENOCR_USE_GPU", "false").lower() in ("1", "true", "yes", "auto")
if GPU is False:
    GPU = False
_lock = threading.Lock()  # 单实例串行推理，防止显存/内存爆掉


def get_engine(model: str):
    if model in _engines:
        return _engines[model]
    from openocr import OpenOCR
    if model == "opendoc-0.1b":
        eng = OpenOCR(task="doc", use_layout_detection=True, use_gpu=GPU)
    elif model == "unirec-0.1b":
        eng = OpenOCR(task="unirec", use_gpu=GPU)
    else:
        raise KeyError(model)
    _engines[model] = eng
    return eng


def _warmup():
    for m in MODELS:
        get_engine(m)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _warmup()          # 启动时加载模型，就绪后 /health 才返回 ok
    yield


app = FastAPI(title="OpenOCR OpenAI-compatible API", version="1.1", lifespan=lifespan)


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


def _extract_images(messages: list[ChatMessage]) -> list[str]:
    """从 messages 中提取所有图片(data URI 或 http url)，返回临时文件路径列表。"""
    import urllib.request
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
                p = os.path.join(tempfile.gettempdir(), f"oocr_{uuid.uuid4().hex}{suffix}")
                urllib.request.urlretrieve(url, p)
                paths.append(p)
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
    text, _ = engine(image_path=img_path, max_length=max_tokens)
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
            "max_image_kb": MAX_IMAGE_BYTES // 1024}


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
        return _err(500, f"推理失败: {e}", "server_error")
    finally:
        for p in images:
            try:
                os.remove(p)
            except OSError:
                pass

    return _answer(req.model, "\n\n---\n\n".join(contents), req.stream)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8089")))