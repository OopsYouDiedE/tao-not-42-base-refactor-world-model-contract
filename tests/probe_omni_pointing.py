#!/usr/bin/env python3
"""标定 VLM 的像素坐标约定与指点精度(Nemotron-3-Nano-Omni)。

为什么需要:控制环里让模型输出"相机转多少度"时它答 -1(符号都错);改成"指哪个像素"
后方向立刻正确。但**不能假设它用哪套坐标系**——本脚本在图上画已知位置的红点,
让模型报坐标,再比较"绝对像素"与"1000x1000 归一化"两种解释的误差。

2026-07-09 实测结论(640x360 帧):
    真值(500,300) -> 说 786,845   归一化还原(503,304)  误差 5.2px
    真值(160, 90) -> 说 250,256   还原(160, 92)        误差 2.2px
    真值(320,270) -> 说 498,756   还原(319,272)        误差 2.5px
    真值(600, 60) -> 说 946,167   还原(605, 60)        误差 5.4px
  ⇒ **1000x1000 归一化**,指点精度 2.2-5.4 px(<1%)。

这条标定是 tests/probe_omni_minecraft_lumine.py 里 `--aim pixel` 与
`pixel_to_delta()` 的依据。换模型/换分辨率时先跑这个。

用法:  python tests/probe_omni_pointing.py --image /path/frame.png
"""
from __future__ import annotations

import argparse
import base64
import io

from openai import OpenAI
from PIL import Image, ImageDraw

MODEL = "nemotron_3_nano_omni"
DOTS = [(500, 300), (160, 90), (320, 270), (600, 60)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--image", default="/workspace/assets/mc_warm_120.png")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    base = Image.open(args.image).convert("RGB")
    w, h = base.size
    print(f"frame {w}x{h}\n")
    print(f"{'true':>12} {'model says':>12} {'err_abs':>9} {'err_1000norm':>13}  verdict")

    for dx, dy in DOTS:
        img = base.copy()
        d = ImageDraw.Draw(img)
        d.ellipse([dx - 9, dy - 9, dx + 9, dy + 9], fill=(255, 0, 0),
                  outline=(255, 255, 255), width=3)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "There is one red dot in this image. Give the pixel "
                                         "coordinates of its centre. Answer with just: x,y"},
            ]}],
            max_tokens=24, temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1},
        )
        txt = (r.choices[0].message.content or "").strip()
        try:
            x, y = [float(v) for v in txt.replace("(", "").replace(")", "").split(",")[:2]]
        except ValueError:
            print(f"{(dx, dy)!s:>12} {txt!r:>12}  unparsed")
            continue
        err_abs = ((x - dx) ** 2 + (y - dy) ** 2) ** 0.5
        nx, ny = x / 1000 * w, y / 1000 * h
        err_norm = ((nx - dx) ** 2 + (ny - dy) ** 2) ** 0.5
        verdict = "1000-NORM" if err_norm < err_abs else "ABSOLUTE"
        print(f"{(dx, dy)!s:>12} {txt:>12} {err_abs:9.1f} {err_norm:13.1f}  {verdict}")


if __name__ == "__main__":
    main()
