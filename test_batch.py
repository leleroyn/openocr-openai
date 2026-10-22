"""批量跑一批图片并落盘识别结果，供 score_batch.py 打分；顺便打印逐张耗时与平均耗时。

用法: python test_batch.py <图片目录|通配符>... [--out ocr_texts.json] [--model scriptmoe] [--exts .jpg,.png]
环境变量: OCR_BASE（默认 http://127.0.0.1:8089/v1）
"""
import base64
import glob
import json
import os
import sys
import time

import requests

BASE = os.environ.get("OCR_BASE", "http://127.0.0.1:8089/v1")
DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def img_part(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/" + ext
    return {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + b64}}


def parse(argv):
    """位置参数全是图片来源（可多个）；输出路径必须走 --out，避免把图片路径当输出把图覆盖掉。"""
    model, exts, pos, out, i = "scriptmoe", list(DEFAULT_EXTS), [], "ocr_texts.json", 0
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            model = argv[i + 1]
            i += 2
        elif a == "--exts":
            exts = [e.strip() for e in argv[i + 1].split(",")]
            i += 2
        elif a == "--out":
            out = argv[i + 1]
            i += 2
        else:
            pos.append(a)
            i += 1
    return model, exts, pos, out

def main(argv):
    model, exts, pos, out = parse(argv)
    if not pos:
        sys.exit(__doc__)
    if not out.lower().endswith(".json"):
        sys.exit("--out 必须是 .json 文件（不给就默认 ocr_texts.json），拒绝写到 " + out)
    files = []
    for src in pos:
        if os.path.isdir(src):
            files += [f for f in glob.glob(os.path.join(src, "*"))
                      if f.lower().endswith(tuple(exts))]
        else:
            files += glob.glob(src)
    files = sorted(set(files))
    if not files:
        sys.exit("没有匹配到图片: " + ", ".join(pos))

    texts, rows = {}, []
    for p in files:
        body = {"model": model, "messages": [{"role": "user", "content": [img_part(p)]}]}
        t0 = time.time()
        r = requests.post(BASE + "/chat/completions", json=body, timeout=1800)
        dt = (time.time() - t0) * 1000
        name = os.path.basename(p)
        if r.status_code != 200:
            print(name.ljust(28) + " HTTP " + str(r.status_code) + " " + r.text[:120], flush=True)
            texts[name] = ""
            continue
        txt = r.json()["choices"][0]["message"]["content"]
        texts[name] = txt
        rows.append({"name": name, "ms": round(dt, 1), "chars": len(txt)})
        print("%-28s %8.0f ms %6d 字" % (name, rows[-1]["ms"], rows[-1]["chars"]), flush=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=1)
    if rows:
        tot = sum(r["ms"] for r in rows)
        print("\n张数=%d 总耗时=%.0fms 平均=%.0fms/张" % (len(rows), tot, tot / len(rows)))
    print("成功 %d/%d，打分: python score_batch.py %s" % (sum(1 for v in texts.values() if v), len(files), out))


if __name__ == "__main__":
    main(sys.argv[1:])
