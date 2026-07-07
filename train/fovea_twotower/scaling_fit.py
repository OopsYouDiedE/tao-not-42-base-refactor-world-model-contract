#!/usr/bin/env python3
"""F2 规模外推:快头容量幂律 + 数据受限判定 + 甜点规模。

模型:err(N) = a·N^(-b) + c。
c 不自由拟合——**固定为观测一致教师的追踪误差 12°**(BC 学生的监督天花板:
学生模仿教师的行为分布,不可能系统性好于教师;v12 实测学生 13.8° vs 教师
13.2° 已贴脸)。拟合点=容量受限区 3 点(2.4/8.8/22M,同数据 83 局),
64M(20.6°,F1 n=27)**排除在幂律外**:n 加倍后反转=数据受限区,幂律前提
(数据充分)不成立——它本身就是"甜点已过"的直接证据。
自由度=1,结论只到量级+方向。

甜点判定(两条独立线索交叉):
  ①实测:容量曲线在 22M 处最后一次可测改善,64M 反转 → 甜点 ≤22M@83局;
  ②外推:err(N)-c 每翻倍容量的边际改善 <1°(运行间方差 3° 的 1/3,
    不可测)处 → N*;
数据侧:22M 已到教师天花板 2° 内 → 继续买容量最多买 2°;**移动天花板本身
(更好教师/GRPO/更多样课程)才是主杠杆**——先加数据/改教师,再谈容量。
"""
import json

import numpy as np
from scipy.optimize import curve_fit

C_TEACHER = 12.0        # 观测一致 token 教师 p1 中位(跨 v7-v14 评测 10.8-13.2°)
POINTS_CAP = [          # 容量受限区(同数据 83 局,每尺寸最优步数)
    (2.4, 21.8, 12),    # (参数量M, 闭环p1°, n局)
    (8.8, 17.8, 12),
    (22.0, 14.15, 26),  # F1 高精度点
]
POINT_DATABOUND = (64.0, 20.6, 27)   # F1:n加倍后反转,不进幂律


def main():
    N = np.array([p[0] for p in POINTS_CAP])
    E = np.array([p[1] for p in POINTS_CAP])
    (a, b), _ = curve_fit(lambda n, a, b: a * n ** (-b) + C_TEACHER, N, E,
                          p0=[15, 0.3], bounds=([0.1, 0.01], [100, 3]),
                          maxfev=20000)
    f = lambda n: a * n ** (-b) + C_TEACHER
    rmse = float(np.sqrt(np.mean((f(N) - E) ** 2)))
    grid = np.logspace(np.log10(2), np.log10(2048), 600)
    marg = -np.gradient(f(grid), np.log2(grid))       # °/倍容量
    n_star = float(grid[np.argmax(marg < 1.0)])       # 边际<1°(方差3°/3)
    out = dict(
        model=f"err(N)=a*N^-b+c, c固定={C_TEACHER}(教师天花板)",
        fit=dict(a=round(a, 2), b=round(b, 3), rmse=round(rmse, 2)),
        points_cap=POINTS_CAP, point_databound=POINT_DATABOUND,
        pred_if_data_sufficient={f"M{int(n)}": round(f(n), 1)
                                 for n in (22, 64, 128, 512)},
        sweet_spot=dict(
            measured="≤22M@83局(64M反转=数据受限直接证据)",
            extrapolated_M=round(n_star, 0),
            note="外推甜点=数据充分假设下边际<1°/倍;当前数据下实测甜点更小"),
        ceiling_gap_at_22M=round(14.15 - C_TEACHER, 2),
        verdict="22M 已在教师天花板 2.2° 内;容量杠杆封顶 ~2°;"
                "主杠杆=移动天花板(GRPO/更好教师/课程多样化)+数据扩容")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open("runs/scaling_fit.json", "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
