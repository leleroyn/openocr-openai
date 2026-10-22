"""new-api 接入自检：模拟网关渠道测试（纯文本）、流式、图片请求，逐条打印 HTTP 码与响应摘要。

用法: python test_newapi.py [图片路径]
环境变量: OCR_BASE（默认 http://127.0.0.1:8089/v1）
"""
import base64
import json
import os
import sys
import time

import requests

BASE = os.environ.get("OCR_BASE", "http://127.0.0.1:8089/v1")
MODEL = "scriptmoe"
REMOVED = "opendoc-0.1b"    # 旧 id：别名已删，必须 404（网关渠道模型名要改成 scriptmoe）


def img_part(path: str) -> dict:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def show(tag: str, r, stream: bool):
    if stream:
        parts, done = [], False
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    done = True
                    continue
                d = json.loads(payload)
                delta = d.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    parts.append(delta["content"])
        text = "".join(parts)
        print(f"{tag:<34} HTTP {r.status_code} | 流式 {'[DONE]OK' if done else '[DONE]缺失'} | {len(text)} 字 | {text[:60]!r}")
    else:
        if r.status_code != 200:
            print(f"{tag:<34} HTTP {r.status_code} | {r.text[:160]}")
            return
        d = r.json()
        text = d["choices"][0]["message"]["content"]
        print(f"{tag:<34} HTTP {r.status_code} | finish={d['choices'][0]['finish_reason']} | {len(text)} 字 | {text[:60]!r}")


def post(tag: str, body: dict, stream: bool = False):
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=body, stream=stream, timeout=1800)
    dt = time.time() - t0
    show(f"{tag} ({dt:.1f}s)", r, stream)


print("health :", requests.get(BASE.replace("/v1", "") + "/health").text)
print("models :", requests.get(f"{BASE}/models").text[:200])

# 1. 网关渠道测试原样请求：content 是纯字符串，无图片
post("1 纯文本(网关测试)", {"model": MODEL, "messages": [{"role": "user", "content": "你好"}]})
# 2. 纯文本 + 流式
post("2 纯文本+流式", {"model": MODEL, "messages": [{"role": "user", "content": "你好"}], "stream": True}, stream=True)
# 3. content 数组只有 text part
post("3 数组仅text", {"model": MODEL, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
# 4. 已删除的旧 id：必须 404，证明别名真的清干净（网关渠道模型名需改成 scriptmoe）
post("4 旧id应404", {"model": REMOVED, "messages": [{"role": "user", "content": "你好"}]})
# 5. 未知模型
post("5 未知模型", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
# 6. 带网关常见附加字段
post("6 附加字段", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                  "temperature": 0.7, "n": 1, "presence_penalty": 0, "seed": 1,
                  "user": "new-api", "response_format": {"type": "text"}})

if len(sys.argv) > 1:
    p = sys.argv[1]
    parts = [img_part(p)]
    # 7. 图片非流式
    post("7 图片非流式", {"model": MODEL, "messages": [{"role": "user", "content": parts}]})
    # 8. 图片流式
    post("8 图片+流式", {"model": MODEL, "messages": [{"role": "user", "content": parts}], "stream": True}, stream=True)
    # 9. 图片 + 文本混合（网关常见）
    post("9 图片+文本", {"model": MODEL, "messages": [{"role": "user", "content": parts + [{"type": "text", "text": "识别"}]}]})