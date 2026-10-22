"""给 test_batch.py 产出的识别结果打分：8 个营业执照关键字段命中率 + 18 位统一社会信用代码校验位。

校验位用 GB32100 规则（前 17 位按 3^i mod 31 加权）验代码是不是合法编号：识别错任意一个字符，校验位几乎
必然过不了。它不依赖人工标注，是证照场景最硬的正确性指标，适合新旧引擎横向比。

用法: python score_batch.py <texts.json> [texts2.json ...]   # 多份并列时逐份打印，便于横向对比
"""
import json
import re
import sys

ORG = r"(?:有限公司|股份有限公司|有限责任公司|集团有限公司|合伙企业|个人独资企业|工厂|厂|商店|商行|经营部|门市部|工作室|中心|店|公司)"
RES = {
    "名称": r"([\u4e00-\u9fff]{2,18}?" + ORG + r")",
    "类型": r"((?:有限责任公司|股份有限公司|有限责任公司[（(][^）)]{2,14}[）)]|个人独资企业|合伙企业|个体工商户|法人商事主体))",
    "法定代表人": r"法\s*定\s*代\s*表\s*人\s*[:：]?\s*([\u4e00-\u9fff]{2,4})",
    "注册资本": r"((?:人民币)?[壹贰叁肆伍陆柒捌玖拾佰仟万亿圆元零]{2,24}(?:万|亿)[圆元整]{0,3}|(?:人民币)?\d+(?:\.\d+)?(?:万|亿)?[圆元]整?)",
    "成立日期": r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
    "住所": r"([\u4e00-\u9fff]{2,10}?(?:省|市|自治区|县|区)[\u4e00-\u9fffA-Za-z0-9（）()\-]{1,50}?(?:号|室|层|幢|栋|楼|铺|区))",
    "代码": r"(?<![0-9A-Z])([0-9A-HJ-NP-Z]{18})(?![0-9A-Z])",
    "经营范围": r"(?:施工|工程|销售|服务|安装|承包|设计|租赁|批发|零售|贸易)",
}
KEYS = list(RES.keys())
FLAT = re.compile(r"[\s\u3000]+")

CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"   # GB32100 用到的 31 个字符（无 I O S Z V）
VAL = dict((c, i) for i, c in enumerate(CHARS))
W = [pow(3, i, 31) for i in range(17)]
CODE18 = re.compile(r"(?=(\d{8}[A-Z0-9]{10}))")


def code_ok(code):
    code = code.upper()
    if len(code) != 18 or any(c not in VAL for c in code):
        return None
    s = sum(VAL[c] * w for c, w in zip(code[:17], W))
    c18 = 31 - (s % 31)
    return CHARS[0 if c18 == 31 else c18] == code[17]


def valid_codes(text):
    seen = []
    for m in CODE18.finditer(text.upper()):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return [c for c in seen if code_ok(c)]


def extract(text):
    f = FLAT.sub("", text)
    out = {}
    for k, pat in RES.items():
        m = re.search(pat, f)
        if not m:
            out[k] = ""
        else:
            out[k] = m.group(1) if m.groups() else "有"
    return out


def main(paths):
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        data = json.load(open(p, encoding="utf-8"))
        n = len(data)
        hit = dict((k, 0) for k in KEYS)
        n_ok = chars = done = 0
        print("== " + p)
        for name in sorted(data):
            txt = data[name]
            if not txt:
                print("  %-28s 请求失败" % name)
                continue
            done += 1
            chars += len(txt)
            e = extract(txt)
            for k in KEYS:
                if e[k]:
                    hit[k] += 1
            ok = bool(valid_codes(txt))
            n_ok += 1 if ok else 0
            print("  %-28s 字段 %d/8  校验位 %s" % (name, sum(1 for k in KEYS if e[k]), "OK" if ok else "-"))
        tot = sum(hit.values())
        print("合计 %d/%d 张：关键字段 %d/%d = %d%%  校验位通过 %d/%d  平均字数 %d" % (
            done, n, tot, done * len(KEYS), round(100.0 * tot / max(done * len(KEYS), 1)), n_ok, done,
            chars // max(done, 1)))
        print("   " + "  ".join("%s %d/%d" % (k, hit[k], done) for k in KEYS))


if __name__ == "__main__":
    main(sys.argv[1:])
