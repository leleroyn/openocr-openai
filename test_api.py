"""接口冒烟测试：模型 scriptmoe × 单图/多图，打印 HTTP 码、耗时与识别结果。

用法: python test_api.py <图片路径> [图片路径...]
环境变量: OCR_BASE（默认 http://127.0.0.1:8089/v1）
"""
import base64
import os
import sys
import time

import requests

BASE = os.environ.get("OCR_BASE", "http://127.0.0.1:8089/v1")
MODELS = ("scriptmoe",)   # 只有一个引擎，模型名必须精确匹配（旧 id 已删）


def img_part(path: str) -> dict:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call(model: str, parts: list) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": parts}]}
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=body, timeout=1800)
    dt = time.time() - t0
    print(f"\n===== {model} | {len(parts)} 张图 | HTTP {r.status_code} | {dt:.1f}s =====")
    if r.status_code != 200:
        print(r.text[:800])
        return
    print(r.json()["choices"][0]["message"]["content"][:1500])


if len(sys.argv) < 2:
    sys.exit(__doc__)

imgs = sys.argv[1:]
print("health :", requests.get(BASE.replace("/v1", "") + "/health").text)
print("models :", requests.get(f"{BASE}/models").text[:300])
for model in MODELS:
    call(model, [img_part(imgs[0])])
if len(imgs) > 1:
    for model in MODELS:
        call(model, [img_part(p) for p in imgs])
