# -*- coding: utf-8 -*-
"""快脑 token 流契约:TokenHead(感知→token)/TokenTeacher(观测一致教师)/goal 相对折叠。

本模块 = 快脑的输入端单一事实源(原散在 tests/integration/collect_track_cmd.py 与
train/fovea_twotower/train_track_cmd.py,被采集/训练/评测/全回路四方共用,归位到 net/)。

token 契约(TokenHead 输出,[K, 6+C+1]):
    [cx, cy, w, h, p_cls, area | softmax 概率 C+1 类]  空间归一化,按 面积×概率 降序取 top-K
goal 相对折叠(goal_relative,[T,K,6+C+1]→[T,K,8]):
    [几何6, p_goal, p_other_max] —— 快头结构上类无关,"听指挥"内建在输入契约;
    切换指令 = p_goal 列换列,策略应立即重定向。

教训沉淀(判据与出处见 docs/architectures/fovea-experiments-index.md):
  · token 必须从 G1 验收的 conv 分割头连通域构建,pf 提案池化命名假阳性泛滥
    (灰墙冒充铁,教师锁假目标 err 89°;G1 量化:提案并集 0.15 vs conv 稠密 0.53);
  · 教师必须是学生观测的函数(TokenTeacher 只消费 token,位姿/raycast 只许评测),
    且必须确定性(搜索方向恒右转——随机方向=不可观测潜变量,BC 条件均值归零,
    v12 切换率钉死 0.17 的根因,确定化后翻倍至 0.32);
  · 右下角=第一人称手持物常驻区必须遮罩(C1b:石镐被认成 dirt,教师原地追自己的手)。
"""
import numpy as np
import torch

CLASSES = ["iron_ore", "coal_ore", "dirt"]   # 感知核心类(G1 校准),全栈单一定义
PARSE_DIM_REL = 8              # goal 相对 token 维度:[几何6, p_goal, p_other_max]
MAX_CAM = 18.0                 # 单步相机增量上限(deg)
REACH_STOP = 2.8               # 到达即停的距离
EYE_H = 1.62                   # MC 眼高(feet→eye)


def as_hwc(rgb):
    """env obs rgb(CHW 或 HWC)→ HWC ndarray(采集/评测/全回路重复片段的单一实现)。"""
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return arr


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def aim_solution(pose, tgt):
    """位姿 + 目标点(前脸中心) → (期望yaw, 期望pitch, 角误差deg, 距离)。

    MC 约定:yaw=0 朝 +z,forward=(-sin y·cos p, -sin p, cos y·cos p)。"""
    x, y, z, yaw, pitch = pose
    eye = np.array([x, y + EYE_H, z])
    v = np.asarray(tgt, float) - eye
    d = float(np.linalg.norm(v))
    vy = v / (d + 1e-9)
    des_yaw = float(np.degrees(np.arctan2(-vy[0], vy[2])))
    des_pitch = float(np.degrees(-np.arcsin(np.clip(vy[1], -1, 1))))
    yr, pr = np.radians(yaw), np.radians(pitch)
    fwd = np.array([-np.sin(yr) * np.cos(pr), -np.sin(pr), np.cos(yr) * np.cos(pr)])
    err = float(np.degrees(np.arccos(np.clip(fwd @ vy, -1, 1))))
    return des_yaw, des_pitch, err, d


def goal_relative(tokens, goal_idx):
    """[T,K,6+C+1] + [T] → [T,K,8]。"""
    T, K, _ = tokens.shape
    geo = tokens[..., :6]
    prob = tokens[..., 6:]                                  # [T,K,C+1]
    pg = np.take_along_axis(prob, goal_idx[:, None, None].repeat(K, 1), 2)  # [T,K,1]
    masked = prob.copy()
    np.put_along_axis(masked, goal_idx[:, None, None].repeat(K, 1), -1, 2)
    pom = masked.max(-1, keepdims=True)
    return np.concatenate([geo, pg, pom], -1).astype(np.float32)


class AimTeacher:
    """位姿投影教师(oracle,有特权只作对照):指令类最近方块前脸中心 → 比例控制相机
    + 对准前进;epsilon 掺随机(覆盖度)。

    教训:目标出视野时它仍能直转目标——学生 token 里此时无方向信息,重获取不可学
    (mamba_seed 特权教师教训的隐蔽复发,E1 v2–v5 连败根因)→ 示范主路用 TokenTeacher。"""

    def __init__(self, rng, epsilon=0.08):
        self.rng, self.eps = rng, epsilon

    def __call__(self, noop, pose, gt_blocks, goal_cls):
        a = dict(noop)
        blocks = gt_blocks[goal_cls]
        cands = [aim_solution(pose, (b[0] + .5, b[1] + .5, b[2])) for b in blocks]
        des_yaw, des_pitch, err, d = min(cands, key=lambda c: c[3])
        if self.rng.random() < self.eps:
            a["camera_yaw"] = float(self.rng.normal(0, 10))
            a["camera_pitch"] = float(self.rng.normal(0, 6))
            a["forward"] = bool(self.rng.random() < 0.3)
            return a, err, d
        a["camera_yaw"] = float(np.clip(0.6 * wrap180(des_yaw - pose[3]),
                                        -MAX_CAM, MAX_CAM))
        a["camera_pitch"] = float(np.clip(0.6 * (des_pitch - pose[4]),
                                          -MAX_CAM, MAX_CAM))
        if err < 10.0 and d > REACH_STOP:
            a["forward"] = True
        return a, err, d


class TokenTeacher:
    """观测一致教师(v6):只消费学生同款 token,不用位姿 oracle → BC 可学性由构造保证。

    行为:goal token 可见(p_goal>τ)→ 瞄准其 (cx,cy)(水平 FOV≈100°/竖直 70° 映射),
    居中且 area<近距阈 → forward;不可见 → 匀速 yaw 搜索(恒右转)+ pitch 缓回,
    贴脸(超大连通域)后退拉开视野。"""
    HFOV, VFOV = 100.0, 70.0
    TAU, AREA_NEAR, CENT = 0.22, 0.038, 0.05   # 0.030 停太远够不着 2.8 格线;0.045 贴脸切换后找不到新目标(v10 教师 p2=86°)
    GAIN, HOLD = 0.5, 4          # 低增益防过冲丢目标;短记忆抗检测闪断

    def __init__(self, rng, epsilon=0.05, area_near=None):
        # area_near 可覆盖:审计发现默认 0.038 使教师停在 ~5.1 格,被到达判据线
        # 2.8 格挡在外面——"学生到达反超教师"疑为参数假象;公平重测用 ~0.10
        if area_near is not None:
            self.AREA_NEAR = float(area_near)
        self.rng, self.eps = rng, epsilon
        self.search_dir = 1.0
        self.last_off, self.hold_left = None, 0

    def new_segment(self):
        # 搜索方向恒右转:随机方向=不可观测潜变量,同观测下标签 ±15° 对冲,
        # BC 条件均值=0 → 学生永远学不会发起搜索(v12 切换率钉死 0.17 的根因)
        self.search_dir = 1.0
        self.last_off, self.hold_left = None, 0

    def __call__(self, noop, toks, goal_idx, pitch_now):
        a = dict(noop)
        if self.rng.random() < self.eps:
            a["camera_yaw"] = float(self.rng.normal(0, 10))
            a["camera_pitch"] = float(self.rng.normal(0, 6))
            return a
        pg = toks[:, 6 + goal_idx] * (toks[:, 4] > 0).astype(np.float32)
        score = pg * toks[:, 5]                         # 面积×概率:大连通域更可信
        j = int(np.argmax(score))                       # (取最高p会咬小噪点)
        off = None
        if pg[j] > self.TAU:
            off = (toks[j, 0] - 0.5, toks[j, 1] - 0.5, toks[j, 5])
            self.last_off, self.hold_left = off, self.HOLD
        elif self.hold_left > 0:                        # 闪断:按记忆位置继续压
            self.hold_left -= 1
            off = self.last_off
        if off is not None:
            offx, offy, area = off
            if abs(offx) > 0.02:                        # 死区防抖
                a["camera_yaw"] = float(np.clip(offx * self.HFOV * self.GAIN,
                                                -MAX_CAM, MAX_CAM))
            if abs(offy) > 0.02:
                a["camera_pitch"] = float(np.clip(offy * self.VFOV * self.GAIN,
                                                  -MAX_CAM, MAX_CAM))
            if abs(offx) < self.CENT and abs(offy) < self.CENT \
                    and area < self.AREA_NEAR:
                a["forward"] = True
        else:                                           # 不可见:搜索
            a["camera_yaw"] = float(15.0 * self.search_dir)
            a["camera_pitch"] = float(np.clip(-0.3 * pitch_now, -8, 8))
            if toks[:, 5].max() > 0.20:                 # 贴脸(超大连通域):后退拉开视野
                a["back"] = True                        # (v10 教训:贴墙切换重获取不可能)
        return a


class TokenHead:
    """G1 验收的 conv 分割头 → 连通域 → 每帧 [K, 6+C+1] token(几何 + 概率)。

    v6 教训:pf 提案池化命名假阳性泛滥(灰墙冒充铁,教师锁假目标 err 89°)——
    G1 已量化:提案并集 0.15 vs conv 稠密 0.53。token 必须从**验收过的分割通道**
    的连通域构建;pf 提案留给开放集,不再承担核心类命名。"""

    def __init__(self, vectors="runs/g1_vectors.pt", K=8, device="cuda",
                 conv_head="runs/g1_conv_head.pt", min_area=150, classes=None):
        import cv2 as _cv2
        from net.fovea_twotower.seg_head import ConvSegHead
        from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
        self.cv2 = _cv2
        self.classes = list(classes) if classes else list(CLASSES)
        self.u = UnifiedYoloe26(device=device, pf_w=None)
        self.pad = pad384
        self.head = ConvSegHead(ncls=len(self.classes) + 1).to(device).eval()
        self.head.load_state_dict(torch.load(conv_head, map_location=device,
                                             weights_only=False))
        self.K, self.D, self.min_area = K, 6 + len(self.classes) + 1, min_area

    @torch.no_grad()
    def __call__(self, rgb_hwc):
        img = self.pad(np.ascontiguousarray(rgb_hwc))
        prob = self.head(self.u.embed(img)[0].float())[0].softmax(0)  # [C+1,384,640]
        lab = prob.argmax(0).cpu().numpy().astype(np.uint8)
        prob_np = prob.cpu().numpy()
        cands = []
        for ci in range(len(self.classes)):
            n, cc, stats, cent = self.cv2.connectedComponentsWithStats(
                (lab == ci).astype(np.uint8), 8)
            for j in range(1, n):
                x, y, w, h, area = stats[j]
                if area < self.min_area:
                    continue
                if cent[j][0] > 0.70 * 640 and cent[j][1] > 0.58 * 384:
                    continue        # 右下角=第一人称手持物/手臂常驻区(C1b 教训:
                                    # 石镐被认成 dirt,教师原地转圈追自己的手)
                m = cc == j
                p = prob_np[:, m].mean(1)                       # [C+1] 域内均值
                cands.append((float(area) * float(p[ci]),
                              [cent[j][0] / 640, cent[j][1] / 384,
                               w / 640, h / 384, float(p[ci]),
                               area / (640 * 384)], p))
        cands.sort(key=lambda c: -c[0])
        toks = np.zeros((self.K, self.D), np.float32)
        for j, (_, geo, p) in enumerate(cands[:self.K]):
            toks[j, :6] = geo
            toks[j, 6:] = p
        return toks
