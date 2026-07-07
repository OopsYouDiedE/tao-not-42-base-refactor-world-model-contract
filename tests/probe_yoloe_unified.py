# -*- coding: utf-8 -*-
"""统一 YOLOE 头对拍探针:重建打分 ≟ native 打分(net/fovea_twotower/yolo_unified.py)。

三项验证(教训应用:探针必须探被下游实际消费的通道,先对拍再谈校准):
  V1 分数重建:钩 cv4 的 (输入嵌入, 输入文本向量, 输出分数),用
     BN(emb)·norm(w)×exp(ls)+bias 重建,max|Δ| 必须 ~0(证明打分数学吃透);
  V2 文本侧复现:text_bank(names) ≟ cv4 实际收到的 w(归一后)——证明校准原型可以
     与文本 PE 同空间互换;
  V3 pf 提案 + 融合 token 流:真实 640×360 校准帧上端到端跑通,存叠加图。

用法: PYTHONPATH=. .venv/bin/python tests/probe_yoloe_unified.py
"""
import glob

import numpy as np
import torch
import torch.nn.functional as F

from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384

NAMES = ["iron ore", "coal ore", "dirt", "stone wall"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)

    z = np.load(sorted(glob.glob("runs/data/calib640/*.npz"))[0], allow_pickle=True)
    img = pad384(z["frames"][10].transpose(1, 2, 0))            # HWC u8 384×640

    # ── V1/V2:钩 cv4 输入输出 ──
    grabbed = []
    hooks = [m.register_forward_hook(lambda _m, inp, out: grabbed.append((inp, out)))
             for m in u.head.cv4]
    u.pm.set_classes(NAMES, u.pm.get_text_pe(NAMES))
    u.embed(img)                        # 直连前向(不融合),同时触发 cv4 钩子
    for h in hooks:
        h.remove()
    assert len(grabbed) == 3, f"期望 3 尺度,得 {len(grabbed)}"

    print("== V1 分数重建(各尺度 max|Δ|)==")
    for i, ((emb, w), out) in enumerate(grabbed):
        bn = u.head.cv4[i].norm(emb)
        wn = F.normalize(w, dim=-1, p=2)
        rec = (torch.einsum("bchw,bkc->bkhw", bn, wn)
               * u.head.cv4[i].logit_scale.exp() + u.head.cv4[i].bias)
        d = (rec - out).abs().max().item()
        print(f"  P{i+3}: max|Δ|={d:.2e} {'PASS' if d < 1e-3 else 'FAIL'}")

    print("== V2 文本侧复现 ==")
    bank = u.text_bank(NAMES)                                   # [C,512]
    w_native = F.normalize(grabbed[0][0][1], dim=-1, p=2)[0].float()
    d = (bank - w_native).abs().max().item()
    print(f"  bank vs native w: max|Δ|={d:.2e} {'PASS' if d < 1e-3 else 'FAIL'}")

    # ── V3:pf 提案 + 融合 token ──
    u.pm.set_classes(["object"], u.pm.get_text_pe(["object"]))  # 复位占位类
    toks, masks = u.forward(img, bank, conf=0.1)
    print(f"== V3 token 流 [{toks.shape[0]},{toks.shape[1]}] "
          f"(几何6 + 类{bank.shape[0]}) ==")
    for j in range(min(6, len(toks))):
        g = toks[j]
        cls = NAMES[int(np.argmax(g[6:]))]
        print(f"  #{j} conf={g[4]:.2f} cx,cy=({g[0]:.2f},{g[1]:.2f}) "
              f"wh=({g[2]:.2f},{g[3]:.2f}) top={cls} cos={g[6:].max():.3f}")
    import os

    import cv2
    os.makedirs("runs/probe_yoloe", exist_ok=True)
    vis = img.copy()
    for j, g in enumerate(toks):
        x1 = int((g[0] - g[2] / 2) * 640); y1 = int((g[1] - g[3] / 2) * 384)
        x2 = int((g[0] + g[2] / 2) * 640); y2 = int((g[1] + g[3] / 2) * 384)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite("runs/probe_yoloe/unified_overlay.png",
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[probe] {len(toks)} 提案叠加 → runs/probe_yoloe/unified_overlay.png")


if __name__ == "__main__":
    main()
