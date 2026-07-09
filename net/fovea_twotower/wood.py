# -*- coding: utf-8 -*-
"""木材链感知扩类(M-IRON 第一里程碑技能;net/fovea_twotower)。

设计定案(理由入档 experiments-index):
- WOOD_CLASSES = CLASSES+["log"],**不动全局 CLASSES**(课程评测/旧demo依赖
  3 类 token 布局;goal_relative 折叠与类数无关→22M 学生零重训插拔,借
  本管线实测该契约);
- 自然树 GT = raycast 扫描累积:相机网格扫掠,命中 translation_key 含
  "log" 的方块坐标入集;世界静态(观察策略不攻击)→全程帧均可投影。
  已知偏差(登记):①未扫到的树干块/视野内其他树=未标注正样本(压召回,
  接受);②树干块任意朝向可见→用 8 角凸包投影(project_block 前脸版仅
  适用课程墙),系统性高估约一个透视侧脸;
- 认证无树负帧:扫描零命中的局=对 log 类无污染的自然负样本(树木在
  calib_nat_neg 中未标注,该目录对 log 训练有毒,v6 排除,靠门守铁 FP)。
"""
import numpy as np

from net.fovea_twotower.seg_head import cam_basis, FOV_V, gt_masks, H, W
from net.fovea_twotower.token_stream import CLASSES, EYE_H
from net.fovea_twotower.yolo_unified import PAD_TOP

WOOD_CLASSES = list(CLASSES) + ["log"]
MINE_HOLD = 45     # 粘性挖掘:命中木头后持续砍这么多 tick(相机锁死),穿过 raycast 闪断
                   # 治"零散触发砍不破"——空手破原木约3s连击,期间准星不能滑走(声明脚手架)


def project_block_hull(bx, by, bz, pose):
    """方块 8 角 → 384×640 pad 坐标点集(任意视向;凸包由调用方做)。"""
    x, y, z, yaw, pitch = pose
    eye = np.array([x, y + EYE_H, z])
    f, r, u = cam_basis(yaw, pitch)
    fy = (H / 2) / np.tan(np.radians(FOV_V) / 2)
    pts = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                p = np.array([bx + dx, by + dy, bz + dz]) - eye
                zf = p @ f
                if zf < 0.3:
                    return None
                px = (p @ r) / zf * fy + W / 2
                py = -(p @ u) / zf * fy + H / 2
                pts.append([px, py + PAD_TOP])
    pts = np.array(pts)
    if (pts[:, 0].max() < 0 or pts[:, 0].min() > W
            or pts[:, 1].max() < 0 or pts[:, 1].min() > 384):
        return None
    return pts


def wood_masks(gt, pose):
    """{cls:[[x,y,z],..]} + 位姿 → {cls: bool mask [384,640]}(WOOD_CLASSES 序)。

    课程类(iron/coal/dirt)用**前脸投影**(seg_head.gt_masks,与 G1/eval 同口径,
    避免 8 角凸包把不可见侧脸算进 GT → 保证扩 log 类后 iron/coal/dirt 训练/评测
    标签与 v4 逐像素一致,回归口径干净);log 自然树任意视向 → 8 角凸包。"""
    import cv2
    out = gt_masks({c: gt.get(c, []) for c in CLASSES}, pose)   # 前脸,3 课程类
    m_log = np.zeros((384, 640), np.uint8)
    for b in gt.get("log", []):
        pts = project_block_hull(*b, pose)
        if pts is None:
            continue
        cv2.fillConvexPoly(m_log, cv2.convexHull(pts.astype(np.int32)), 1)
    out["log"] = m_log.astype(bool)
    return out


def wood_label_img(gt, pose):
    """{cls:[[x,y,z],..]} → 逐像素标签 [384,640](WOOD_CLASSES 序,背景=C)。

    log 最后绘制(自然树与课程墙不共域;若碰撞 log 优先)。"""
    ms = wood_masks(gt, pose)
    lab = np.full((384, 640), len(WOOD_CLASSES), np.int64)
    for k, c in enumerate(WOOD_CLASSES):
        lab[ms[c]] = k
    return lab
