#!/usr/bin/env python3
"""Nemotron-3-Nano-Omni NVFP4 单卡 5090 实测探针。

回答两个问题:
  1. 官方 NVFP4 权重能否在 sm_120 上被 vLLM 原生加载(不做任何反量化/转换)?
  2. 4bit 之后效果是否还能用?

方法论说明:BF16(62GB)/FP8(33GB) 都塞不进 32GB,**本机做不了同模型跨精度对照**。
因此质量判据不依赖 baseline,改用**有标准答案的客观任务**:
  - ASR:jfk.wav 原文已知 ⇒ 可算 WER
  - OCR:自渲染字符串 ⇒ 可算精确匹配
  - 视觉语义:Crafter 帧 ⇒ 人工核对是否说得出画面里真实存在的实体
再叠加延迟/显存,对照 knowledge/design_llm_deep_integration.md §1 的 0.5-2s 慢系统预算。

用法:
    bash tests/serve_omni_nvfp4.sh          # 另一个终端
    python tests/probe_omni_nvfp4.py --out docs/results/omni_nvfp4_5090.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

from openai import OpenAI

MODEL = "nemotron_3_nano_omni"
# jfk.wav 的公认原文(whisper.cpp samples/jfk.wav)
JFK_REF = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)
OCR_REF = "SUBGOAL: collect_wood x3 THEN place_table"


# ---------------------------------------------------------------- 工具

def wer(ref: str, hyp: str) -> float:
    """词错误率(Levenshtein / 参考词数)。"""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def b64_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def gpu_mem_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().split("\n")[0])


# ---------------------------------------------------------------- 素材

def make_ocr_image(path: Path) -> None:
    """渲染一张已知字符串的图,用于 OCR 精确匹配。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (720, 160), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 60), OCR_REF, fill="black")
    img.save(path)


def make_crafter_frame(path: Path) -> str | None:
    """一帧真实 Crafter 观测(本仓 net/dreamerv3 的训练域)。

    crafter 会拽 numpy/imageio 等依赖,和 vllm 的 venv 有冲突风险,故**预渲染**:
        python -m venv /workspace/venv-crafter && /workspace/venv-crafter/bin/pip install crafter
        /workspace/venv-crafter/bin/python -c "\
import crafter,numpy as np;from PIL import Image;\
env=crafter.Env(area=(64,64),view=(9,9),size=(256,256),seed=7);obs=env.reset();\
rng=np.random.default_rng(0);\
[env.step(int(rng.integers(env.action_space.n))) for _ in range(6)];\
Image.fromarray(np.asarray(env.render(),dtype='uint8')).save('$PATH')"

    seed=7 + 6 步 rng(0) 的地面真相:满屏草地、玩家居中朝下、右侧 3 棵树、无水无石。
    """
    if path.exists():
        return str(path)
    return None


# ---------------------------------------------------------------- 探针

def ask(client: OpenAI, content, *, max_tokens=512, temperature=0.6,
        thinking=True) -> dict:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    dt = time.perf_counter() - t0
    msg = r.choices[0].message
    return {
        "text": (msg.content or "").strip(),
        "reasoning": (getattr(msg, "reasoning_content", None) or "").strip(),
        "latency_s": round(dt, 2),
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "decode_tok_s": round(r.usage.completion_tokens / dt, 1) if dt else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--assets", default="/workspace/assets")
    ap.add_argument("--out", default="docs/results/omni_nvfp4_5090.json")
    args = ap.parse_args()

    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    results: dict = {"vram_mb_after_load": gpu_mem_mb(), "probes": {}}

    # --- P1 文本推理(thinking mode) -----------------------------------
    print("[P1] text reasoning ...", flush=True)
    r = ask(client, "A farmer has 17 sheep. All but 9 run away. How many are left? "
                    "Answer with just the number.", max_tokens=2048)
    r["correct"] = "9" in r["text"]
    results["probes"]["p1_text_reasoning"] = r
    print(f"     -> {r['text'][:80]!r} correct={r['correct']} {r['decode_tok_s']} tok/s")

    # --- P2 OCR 精确匹配 ------------------------------------------------
    print("[P2] OCR exact-match ...", flush=True)
    ocr_png = assets / "ocr.png"
    make_ocr_image(ocr_png)
    r = ask(client, [
        {"type": "image_url", "image_url": {"url": b64_url(ocr_png, "image/png")}},
        {"type": "text", "text": "Transcribe the text in this image verbatim. "
                                 "Output only the text."},
    ], max_tokens=256, thinking=False, temperature=0.2)
    r["reference"] = OCR_REF
    r["exact_match"] = normalize(r["text"]) == normalize(OCR_REF)
    results["probes"]["p2_ocr"] = r
    print(f"     -> {r['text']!r} exact={r['exact_match']}")

    # --- P3 Crafter 帧 -> 子目标(项目域) --------------------------------
    print("[P3] crafter frame -> subgoal ...", flush=True)
    frame = make_crafter_frame(assets / "crafter.png")
    if frame:
        r = ask(client, [
            {"type": "image_url", "image_url": {"url": b64_url(Path(frame), "image/png")}},
            {"type": "text", "text": "This is a frame from the Crafter game (a 2D Minecraft-like "
                                     "survival game, top-down view, player at center). "
                                     "List the terrain types and objects you can see, then "
                                     "propose one concrete next subgoal for the player."},
        ], max_tokens=1024)
        results["probes"]["p3_crafter_subgoal"] = r
        print(f"     -> {r['text'][:160]!r}")
    else:
        results["probes"]["p3_crafter_subgoal"] = {"skipped": "crafter not installed"}
        print("     -> skipped (crafter not installed)")

    # --- P4 ASR / WER ---------------------------------------------------
    print("[P4] ASR WER ...", flush=True)
    wav = assets / "sample.wav"
    if wav.exists():
        r = ask(client, [
            {"type": "audio_url", "audio_url": {"url": b64_url(wav, "audio/wav")}},
            {"type": "text", "text": "Transcribe the speech verbatim. Output only the transcript."},
        ], max_tokens=256, thinking=False, temperature=0.2)
        r["reference"] = JFK_REF
        r["wer"] = round(wer(JFK_REF, normalize(r["text"])), 4)
        results["probes"]["p4_asr"] = r
        print(f"     -> {r['text']!r}\n        WER={r['wer']}")
    else:
        results["probes"]["p4_asr"] = {"skipped": "no sample.wav"}

    # --- P5 解码吞吐(长生成) --------------------------------------------
    print("[P5] decode throughput ...", flush=True)
    r = ask(client, "Write a 400-word essay about rivers.", max_tokens=600, thinking=False)
    results["probes"]["p5_throughput"] = r
    print(f"     -> {r['decode_tok_s']} tok/s ({r['completion_tokens']} tok in {r['latency_s']}s)")

    results["vram_mb_peak"] = gpu_mem_mb()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
