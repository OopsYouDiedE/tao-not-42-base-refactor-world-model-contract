# -*- coding: utf-8 -*-
"""fovea-twotower 评估域的统计工具(AUC / 配对置信区间 / 脊回归 R²)。

对外接口:
    auc(y, s) — 二分类 ROC-AUC(单类别→nan)。
    paired_ci(fn_a, fn_b, N, boot, seed) — 配对指标差 bootstrap CI。
    ridge_r2(Xtr, Ytr, Xte, Yte) — 多输出脊回归方差加权 R²(+逐锚点 SE/SST)。
    paired_r2_ci(se_a, se_b, sst, boot, seed) — 脊回归 R² 差 bootstrap CI。
    pool9 — 从 L1 blocks 重导出(评估域空间降维统一入口)。
"""
import numpy as np

from blocks import pool9  # noqa: F401  L1 算子,评估域统一从此处取用


def auc(y, s):
    """二分类 ROC-AUC。y/s 任意 shape(ravel);单类别→nan。"""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y).ravel(), np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def paired_ci(fn_a, fn_b, N, boot=500, seed=0):
    """配对指标差 (fn_a-fn_b) 的 bootstrap 95% CI;fn 接受重采样索引。"""
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(boot):
        i = rng.integers(0, N, N)
        d = fn_a(i) - fn_b(i)
        if np.isfinite(d):
            ds.append(d)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi)


def ridge_r2(Xtr, Ytr, Xte, Yte):
    """多输出 Ridge(alpha=10,标准化)。返回 (方差加权 R², 逐锚点 SE, SST)。"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=10.0).fit(sc.transform(Xtr), Ytr)
    pred = rg.predict(sc.transform(Xte))
    se = ((pred - Yte) ** 2).sum(1)                    # [N]
    sst = ((Yte - Yte.mean(0)) ** 2).sum(1)
    return 1 - se.sum() / sst.sum(), se, sst


def paired_r2_ci(se_a, se_b, sst, boot=500, seed=0):
    """两臂脊回归 R² 差 (a-b) 的 bootstrap 95% CI(共用 SST 分母重采样)。"""
    rng = np.random.default_rng(seed)
    N = len(sst)
    ds = []
    for _ in range(boot):
        i = rng.integers(0, N, N)
        ds.append((1 - se_a[i].sum() / sst[i].sum())
                  - (1 - se_b[i].sum() / sst[i].sum()))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi)
