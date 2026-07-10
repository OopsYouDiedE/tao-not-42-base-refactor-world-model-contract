# -*- coding: utf-8 -*-
"""v2 塔(TokenPolicyTower)的训练侧装配:DINO patch + 地图 + 语言 token(设计文档 §7)。

对外接口:
    V2Config     — v2 装配参数(dataclass)
    DinoFrontend — 冻结 DINO 骨干 → 逐帧 patch token(tests 可用同协议 mock 替换)
    V2Policy     — 可学参数容器(TokenPolicyTower + MapWriter + MapReader,单一 state_dict)
    V2Runtime    — 单 episode 状态机:里程计 / 地图写读 / 钉点 / 逐 tick token 记录
    v2_replay    — 更新端按记录 token 重算 logits(与采样同分布)

机制:每 tick 冻结 DINO 出 patch 网格;patch 中心 uv 经 ipm_ground 以自标定
yaw/pitch/FOV 稠密落地,写入北锚定 EgoMapClip(MapWriter);MapReader 读出
grid²·levels 个地图 token,与 patch token(帧堆叠 S=n_frames)、subgoal UTF-8 字节
token、prev 动作 token 一起进 goal-as-query cross-attention。动作头口径消费
action_contract 单一定义,与 v1 逐字一致。

符号标定与降级(自标定立场,net/calibration.py):
  · cmd→旋向符号 = SelfCalib.yaw_sign / pitch_sign(光流增益符号,纯观测,几何普适);
  · 里程计走部署纯净口径:yaw/pitch=相机命令积分×实测符号,平移=键位×自标定步速
    (env pose 只在训练侧标步速,特权不进部署回路);
  · 任一要件测不出(sign / fov / step_blocks 为 None)⇒ 对应通路显式降级:
    不写图、不钉点、平移账本置零;地图 token 退化为常量位置编码,
    physics_vector 有效位为 0——不编数。

梯度口径(如实):GRPO 更新按记录 token 回放,梯度流经 tower 全部参数
(vis_in / map_in / lang_emb / geo_in / xattn / heads);MapWriter.w_c 与
MapReader.proj 需要同图重放写读才有梯度(BC 阶段做,见 map_io.MapReader
docstring),GRPO 路径记录值当常量,不更新它们。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from net.backbone import load_backbone
from net.calibration import SelfCalib
from net.fovea_twotower.ego_map import EgoMapClip
from net.map_io import AimPin, MapReader, MapWriter
from net.token_tower import TokenTowerConfig, build_token_tower, encode_utf8
from train.craftground.action_contract import CAM_BINS, CAM_MAX_DEG, V2_KEYS

# DINO 输入尺寸(须被 patch 整除;90x160 帧就近重采样,uv 相对坐标不变)
_DINO_HW = {"dinov3": (96, 160), "dinov2": (98, 154)}
_PX_MEAN = (0.485, 0.456, 0.406)
_PX_STD = (0.229, 0.224, 0.225)


@dataclass
class V2Config:
    """v2 装配参数。地图尺寸沿 EgoMapClip 预登记配置(size=32/half=32/levels=3)。"""

    dino: str = "dinov3"      # 骨干(用户裁决 dinov3 优先;gated 权重经 HF_TOKEN)
    n_frames: int = 2         # 视觉帧堆叠 S(D1:速度可观测;长程记忆归地图/慢塔)
    d: int = 256
    map_c: int = 8
    map_size: int = 32
    map_half: float = 32.0
    map_levels: int = 3
    map_grid: int = 4         # 地图 token 数 = grid²·levels = 48
    pin_ttl: int = 200        # aim 钉 TTL(tick;=10s@20tps)


class DinoFrontend:
    """冻结 DINO 骨干 → 逐帧 patch token。

    encode: [H,W,3] float32 ∈[0,1] → [Np, enc_dim] float32(no_grad,设备上)。
    属性:enc_dim、n_tokens=Np、uv [Np,2](patch 中心归一屏幕坐标,(0,0)=左上)。
    tests 的 mock 按同协议注入 V2Runtime(AGENTS §2:mock 只活在 tests/)。
    """

    def __init__(self, kind: str, device: str, repo: str | None = None):
        module, patch, dim, n_reg = load_backbone(kind, repo)
        self.bb = module.eval().to(device)
        for p in self.bb.parameters():
            p.requires_grad_(False)
        self.enc_dim, self.n_reg, self.device = dim, n_reg, device
        self.in_hw = _DINO_HW[kind]
        gh, gw = self.in_hw[0] // patch, self.in_hw[1] // patch
        self.n_tokens = gh * gw
        u = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
        v = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], -1).to(device)  # [Np,2]
        self._mean = torch.tensor(_PX_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_PX_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, img: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32,
                            device=self.device).permute(2, 0, 1)[None]
        x = F.interpolate(x, self.in_hw, mode="bilinear", align_corners=False)
        t = self.bb(pixel_values=(x - self._mean) / self._std).last_hidden_state
        return t[0, 1 + self.n_reg:].float()                 # [Np, enc_dim]


class V2Policy(torch.nn.Module):
    """v2 可学参数容器:tower + 地图写读投影,单一 state_dict(checkpoint 单位)。"""

    def __init__(self, vcfg: V2Config, enc_dim: int):
        super().__init__()
        assert CAM_BINS == 11 and len(V2_KEYS) == 20         # 契约由训练端断言(AGENTS §8)
        self.vcfg = vcfg
        tcfg = TokenTowerConfig(d=vcfg.d, vis_dim=enc_dim, n_frames=vcfg.n_frames,
                                n_keys=len(V2_KEYS), camera_bins=CAM_BINS)
        self.tcfg = tcfg
        self.tower = build_token_tower(tcfg)
        self.map_writer = MapWriter(enc_dim, c=vcfg.map_c)
        self.map_reader = MapReader(c=vcfg.map_c, d_out=tcfg.map_dim, grid=vcfg.map_grid)


class V2Runtime:
    """单 episode 状态机(采样端)。

    调用顺序:begin(calib, init_cmd_deg) → 逐 tick tick(small, prev) →
    慢塔刷新 tick 上 on_slow(rep) → episode 末 export() 并入 roll dict。
    prev [22] = [已执行相机度/CAM_MAX_DEG(2) ⊕ 键位(20)],与 grpo_pixel.rollout 同构。
    """

    def __init__(self, policy: V2Policy, frontend, device: str):
        self.p, self.fe, self.device = policy, frontend, device
        self._i_fwd = V2_KEYS.index("forward")
        self._i_back = V2_KEYS.index("back")
        self._i_left = V2_KEYS.index("left")
        self._i_right = V2_KEYS.index("right")

    def begin(self, calib: SelfCalib, init_cmd_deg=(0.0, 0.0)) -> None:
        """episode 初始化。init_cmd_deg = 标定期已发相机命令净和(yaw,pitch,度)。"""
        vc = self.p.vcfg
        self.calib = calib
        ys, ps = calib.yaw_sign, calib.pitch_sign
        self.pose_ok = ys is not None and ps is not None
        self.yaw_deg = (init_cmd_deg[0] * (ys or 0)) % 360.0     # 地图系,0=episode 初始朝向
        self.pitch_deg = float(np.clip(init_cmd_deg[1] * (ps or 0), -90, 90))
        self.map = EgoMapClip(c=vc.map_c, size=vc.map_size, half=vc.map_half,
                              levels=vc.map_levels, device=self.device)
        self.pin = AimPin(ttl_ticks=vc.pin_ttl)
        self.lang = encode_utf8(["explore"])[0]
        self.aim_uv = (0.5, 0.5)
        self._pending_pin = False
        self.hist: list[torch.Tensor] = []
        self.rec: dict[str, list] = dict(vis=[], map=[], lang=[], geo=[])

    # ── 部署纯净里程计:命令积分 + 键位×步速 ────────────────
    def _odometry(self, deg: np.ndarray, keys: np.ndarray) -> tuple[float, float]:
        """记账 a_{t-1}:更新 yaw/pitch 积分,返回世界位移 (east, north)(格)。"""
        if not self.pose_ok:
            return 0.0, 0.0
        self.yaw_deg = (self.yaw_deg + float(deg[0]) * self.calib.yaw_sign) % 360.0
        self.pitch_deg = float(np.clip(
            self.pitch_deg + float(deg[1]) * self.calib.pitch_sign, -90, 90))
        sb = self.calib.step_blocks
        if sb is None:
            return 0.0, 0.0
        f = float(keys[self._i_fwd]) - float(keys[self._i_back])
        r = float(keys[self._i_right]) - float(keys[self._i_left])
        if f == 0.0 and r == 0.0:
            return 0.0, 0.0
        yaw = math.radians(self.yaw_deg)
        east = sb * (f * math.sin(yaw) + r * math.cos(yaw))
        north = sb * (f * math.cos(yaw) - r * math.sin(yaw))
        return east, north

    def on_slow(self, rep: dict) -> None:
        """慢塔刷新:语言 token 换血 + aim 记录 + 登记待钉点(B1:aim 下发即钉图)。

        钉点动作推迟到同 tick 的 tick() 里、里程计记账之后执行——保证钉点用的是
        本 tick 的位姿,不差一步账。
        """
        self.lang = encode_utf8([rep["subgoal"] or "explore"])[0]
        self.aim_uv = (rep["aim"][0] / 1000.0, rep["aim"][1] / 1000.0)
        self._pending_pin = True

    def _geo(self) -> np.ndarray:
        """数值 goal [16]:aim_uv(2) ⊕ 钉点 xy/half+age(3) ⊕ physics_vector(10) ⊕ 备用(1)。"""
        xy, age = self.pin.get()
        half = self.p.vcfg.map_half
        pin3 = ([float(xy[0]) / half, float(xy[1]) / half,
                 age / self.p.vcfg.pin_ttl] if xy is not None else [0.0, 0.0, 0.0])
        return np.concatenate([[self.aim_uv[0], self.aim_uv[1]], pin3,
                               self.calib.physics_vector(), [0.0]]).astype(np.float32)

    def tick(self, small: np.ndarray, prev: np.ndarray):
        """采样端一步(no_grad):记账→写图→读图→组 token→logits。

        small [H,W,3] float32 ∈[0,1];prev [22](a_{t-1},首 tick 全零)。
        返回 (cam_logits [n_mouse,bins], key_logits [n_keys]),未除温度。
        """
        deg = prev[:2] * CAM_MAX_DEG
        dpos = self._odometry(deg, prev[2:])
        self.map.step(dpos)
        self.pin.step(dpos)
        feats = self.fe.encode(small)                        # [Np, D]
        fov = self.calib.fov_y_deg
        if self._pending_pin:                                # on_slow 登记的钉点,用本 tick 位姿
            self._pending_pin = False
            if self.pose_ok and fov is not None:
                self.pin.set(self.aim_uv, yaw=math.radians(self.yaw_deg),
                             pitch=math.radians(self.pitch_deg), fov_y_deg=fov)
        with torch.no_grad():
            if self.pose_ok and fov is not None:             # 稠密落地写图
                self.p.map_writer(self.map, self.fe.uv, feats,
                                  yaw=math.radians(self.yaw_deg),
                                  pitch=math.radians(self.pitch_deg), fov_y_deg=fov)
            map_toks = self.p.map_reader(self.map)           # [K, map_dim]
        self.hist.append(feats)
        if len(self.hist) < self.p.tcfg.n_frames:            # 开局首帧填充(与 v1 同法)
            self.hist = [feats] * (self.p.tcfg.n_frames - len(self.hist)) + self.hist
        self.hist = self.hist[-self.p.tcfg.n_frames:]
        vis = torch.cat(self.hist, dim=0)                    # [S·Np, D] 旧→新
        geo = self._geo()
        self.rec["vis"].append(vis.cpu().numpy())
        self.rec["map"].append(map_toks.cpu().numpy())
        self.rec["lang"].append(self.lang.numpy().copy())
        self.rec["geo"].append(geo)
        with torch.no_grad():
            cam_l, key_l = self.p.tower(
                vis[None], map_toks[None], self.lang[None].to(self.device),
                torch.from_numpy(geo)[None].to(self.device),
                torch.from_numpy(prev.astype(np.float32))[None].to(self.device))
        return cam_l[0], key_l[0]

    def export(self) -> dict:
        """episode 记录 → roll dict 增量(fp32,更新端与采样端数值逐位一致)。"""
        return dict(vis_toks=np.stack(self.rec["vis"]),      # [T, S·Np, D]
                    map_toks=np.stack(self.rec["map"]),      # [T, K, map_dim]
                    lang_toks=np.stack(self.rec["lang"]),    # [T, L] int64
                    geo=np.stack(self.rec["geo"]))           # [T, 16]


def v2_replay(policy: V2Policy, r: dict, sl: slice, device: str):
    """更新端:按记录 token 重算 logits(train 模式,与采样同分布;修复①③口径)。

    返回 (cam_logits [B,n_mouse,bins], key_logits [B,n_keys]),未除温度。
    """
    vis = torch.from_numpy(r["vis_toks"][sl]).to(device)
    map_t = torch.from_numpy(r["map_toks"][sl]).to(device)
    lang = torch.from_numpy(r["lang_toks"][sl]).to(device)
    geo = torch.from_numpy(r["geo"][sl]).to(device)
    prev = torch.from_numpy(r["prevs"][sl]).to(device)
    return policy.tower(vis, map_t, lang, geo, prev)
