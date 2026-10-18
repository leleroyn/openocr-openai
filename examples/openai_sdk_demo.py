"""官方 openai SDK 调用示例。

用法:
    python examples/openai_sdk_demo.py <图片路径> [图片路径...]
    OCR_BASE=http://127.0.0.1:8089/v1 python examples/openai_sdk_demo.py page.png
"""
import base64
import os
import sys

from openai import OpenAI

BASE = os.environ.get("OCR_BASE", "http://127.0.0.1:8089/v1")
MODEL = os.environ.get("OCR_MODEL", "opendoc-0.1b")


def img_part(path: str) -> dict:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


if len(sys.argv) < 2:
    sys.exit(__doc__)

client = OpenAI(base_url=BASE, api_key="dummy")
print("models:", [m.id for m in client.models.list().data])

parts = [{"type": "text", "text": "解析这张图"}] + [img_part(p) for p in sys.argv[1:]]
r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": parts}])
print("--- " + MODEL + " (" + str(len(sys.argv) - 1) + " 图) ---")
print(r.choices[0].message.content)
