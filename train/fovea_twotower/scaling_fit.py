#!/usr/bin/env python3
"""F2 规模外推:快头尺寸阶梯拟合幂律,推甜点规模+数据平衡点。

模型:err(N) = a·N^(-b) + c(N=参数量;c=该数据/协议下的不可约底,
含教师噪声+环境随机+口径地板)。四点拟合三参,自由度=1——结论只到
"量级+方向",不宣称精确点位(已在文档如实标注)。
数据平衡:Chinchilla 式 tokens/param 比在 BC 里对应 帧/参数;
用 C2 示范量曲线(22M 头 125→500 局未饱和)+C3 吞吐(5万局/天)推
"多大的头能被单机数据引擎喂饱"。
"""
import json
import sys

import numpy as np
from scipy.optimize import curve_fit

# (参数量 M, 闭环追踪中位角误差°, n局) —— 每尺寸取最优步数档
# 22M/64M 用 F1 32局高精度点(命令行传入覆盖),其余用 C2 首批
POINTS_DEFAULT = [
    (2.4, 21.8, 12),
    (8.8, 17.8, 12),
    (22.0, 14.5, 12),
    (64.0, 11.4, 12),
]


def powerlaw(n, a, b, c):
    return a * np.power(n, -b) + c


def main():
    pts = POINTS_DEFAULT
    if len(sys.argv) > 1:            # scaling_fit.py 22:12.9 64:11.2
        override = dict(kv.split(":") for kv in sys.argv[1:])
        pts = [(n, float(override.get(str(int(n)), e)), k) for n, e, k in pts]
    N = np.array([p[0] for p in pts])
    E = np.array([p[1] for p in pts])
    (a, b, c), _ = curve_fit(powerlaw, N, E, p0=[30, 0.5, 8],
                             bounds=([0, 0.01, 0], [200, 3, 20]), maxfev=20000)
    pred = powerlaw(N, a, b, c)
    resid = float(np.sqrt(np.mean((pred - E) ** 2)))
    # 甜点=边际收益跌破噪声:dErr/d(log2 N) < 0.5°(≈运行间方差 3° 的 1/6,
    # 即翻倍容量买不到可测改善);同时报 90%/95% 逼近不可约底的规模
    grid = np.logspace(np.log10(2), np.log10(4096), 400)
    err_g = powerlaw(grid, a, b, c)
    dg = -np.gradient(err_g, np.log2(grid))          # °/倍容量
    sweet = grid[np.argmax(dg < 0.5)] if (dg < 0.5).any() else float("inf")
    gap0 = powerlaw(2.4, a, b, c) - c
    n90 = grid[np.argmax((err_g - c) < 0.10 * gap0)]
    out = dict(fit=dict(a=round(a, 2), b=round(b, 3), c_floor=round(c, 2),
                        rmse=round(resid, 2)),
               points=[dict(M=n, err=e, n_eps=k) for n, e, k in pts],
               pred=dict(M128=round(powerlaw(128, a, b, c), 1),
                         M256=round(powerlaw(256, a, b, c), 1),
                         M512=round(powerlaw(512, a, b, c), 1),
                         M1024=round(powerlaw(1024, a, b, c), 1)),
               sweet_spot_M=round(float(sweet), 0),
               reach90pct_floor_M=round(float(n90), 0))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open("runs/scaling_fit.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
