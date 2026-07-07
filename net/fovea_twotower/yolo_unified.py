# -*- coding: utf-8 -*-
"""YOLOE-26 统一解析头:pf 端到端提案 + promptable 区域嵌入 + 可插拔向量库点名。

动机(knowledge/design_fovea_yolo_fasttower.md §2):YOLOE-26 的两项能力要整合进一个体系——
  ① prompt-free(LRPC 头):端到端圈出可能物体(空间分割已零样本迁移到 Minecraft);
  ② promptable(cv3/cv4 对比头):任意向量(文本 PE / 域内校准原型)零样本点名类别。
两 checkpoint 骨干**非同权**(pf 是轻微调副本,除首层外 cos≈1.0000/相对误差 1–3%,
已实测),故不做嫁接:统一模块内两通路各跑其训练工况,在接口层融合——

    propose(imgs)            → pf 提案 [N,(box,conf,mask)]     (① 的能力)
    embed(imgs)              → promptable 三尺度嵌入图(post-BN)  (② 的底料)
    proposal_embed(...)      → 每提案 512d 单位嵌入(FPN 尺度指派+中心双线性采样)
    text_bank(names)         → 文本 PE 经 reprta+归一 → 向量库    (② 零样本口径)
    forward(imgs, bank)      → token 流:几何 + conf + 各类余弦分数 (对快塔的输出契约)

打分数学(与 BNContrastiveHead 逐项对齐,tests/probe_yoloe_unified.py 对拍验证):
    native score_i = BN_i(cv3_i(feat)) · L2norm(reprta(pe)) × exp(logit_scale_i) + bias_i
    本模块用余弦口径:ê = BN_i(emb)/‖·‖,score_cos = ê·bank^T ∈ [-1,1]——跨尺度可比,
    校准原型(域内 ê 均值再归一)与文本 PE 同居一个空间,即"调向量"的落点。

预处理约定:恒等 letterbox——640×360 上下各垫 12px 灰边到 640×384(stride 32 整除),
增益=1,盒/掩膜/GT 坐标全程同一坐标系,无缩放误差。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOP = 12                    # 640×360 → 640×384
STRIDES = (8, 16, 32)           # P3/P4/P5


def cv2_resize_bool(m: np.ndarray, w: int, h: int) -> np.ndarray:
    """bool 掩膜最近邻缩放到 [h,w]。"""
    import cv2
    return cv2.resize(m.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST) > 0


def pad384(img_hwc: np.ndarray) -> np.ndarray:
    """[360,640,3] u8 → [384,640,3] u8(上下 12px 灰 114,同 ultralytics letterbox 填充)。"""
    assert img_hwc.shape[:2] == (360, 640), f"期望 360×640,得 {img_hwc.shape}"
    out = np.full((384, 640, 3), 114, np.uint8)
    out[PAD_TOP:PAD_TOP + 360] = img_hwc
    return out


class UnifiedYoloe26(nn.Module):
    """两通路统一封装(全程 no_grad;下游把输出当固定观测)。"""

    def __init__(self, prompt_w="runs/checkpoints/yoloe-26l-seg.pt",
                 pf_w="runs/checkpoints/yoloe-26l-seg-pf.pt",
                 device="cuda", max_det=64):
        super().__init__()
        from ultralytics import YOLOE
        self.dev, self.max_det = device, max_det
        self.pm = YOLOE(prompt_w)
        self.pf = YOLOE(pf_w) if pf_w else None
        self.head = self.pm.model.model[-1]            # YOLOESegment26
        # cv3 各尺度嵌入图钩子(promptable 前向时填充)
        self._emb_maps = [None] * len(self.head.cv3)
        for i, m in enumerate(self.head.cv3):
            m.register_forward_hook(self._mk_hook(i))
        # predict 需要类集才能前向;设占位类(嵌入钩子与类集无关)
        self.pm.set_classes(["object"], self.pm.get_text_pe(["object"]))

    def _mk_hook(self, i):
        def h(_m, _inp, out):
            self._emb_maps[i] = out.detach()
        return h

    # ── ② promptable 通路 ────────────────────────────────────────────
    @torch.no_grad()
    def text_bank(self, names) -> torch.Tensor:
        """类名列表 → 向量库 [C,512](L2 归一)。

        对拍结论(probe V2):get_text_pe 输出已是 native 文本侧终态——set_classes
        存的向量原样进 cv4,reprta 只作用于推理期新传的 tpe,这里不再套。"""
        pe = self.pm.get_text_pe(names).to(self.dev)          # [1,C,512]
        return F.normalize(pe, dim=-1, p=2)[0].float()

    @torch.no_grad()
    def embed(self, img_hwc: np.ndarray):
        """单帧(384×640 已 pad)→ 三尺度 post-BN 嵌入图 [1,512,H_i,W_i]。

        注意:不走 .predict()——predictor 会把文本向量融合进卷积(is_fused),
        融合后 cv3 尾层变类卷积、嵌入通道消失。直接调底层 nn 前向,不触发融合。"""
        assert not self.head.is_fused, "promptable 头已被融合,嵌入通道不可用(勿对 pm 调 predict)"
        self._emb_maps = [None] * len(self.head.cv3)
        t = (torch.from_numpy(np.ascontiguousarray(img_hwc)).to(self.dev)
             .permute(2, 0, 1)[None].float() / 255.0)
        self.pm.model.to(self.dev).eval()(t)
        assert all(m is not None for m in self._emb_maps), "cv3 钩子未填充"
        return [self.head.cv4[i].norm(self._emb_maps[i].float())
                for i in range(len(self._emb_maps))]

    @staticmethod
    def _assign_scale(w, h):
        """FPN 尺度指派:按盒边长 sqrt(wh) → P3(<64)/P4(<160)/P5。"""
        s = float(np.sqrt(max(w * h, 1.0)))
        return 0 if s < 64 else (1 if s < 160 else 2)

    @torch.no_grad()
    def proposal_embed(self, emb_maps, boxes_xyxy, masks=None, si=0) -> torch.Tensor:
        """提案掩膜池化单位嵌入(P3 细尺度)。boxes [N,4] (+masks [N,384,640]) → [N,512]。

        教训(G1 首轮诊断):盒中心单点采样让"整墙大提案中心恰落在矿上"系统性冒名
        ——36 帧最优提案仅 1 次命名正确。改为掩膜内逐格归一嵌入均值再归一;
        无掩膜退化为盒内池化。统一 P3:跨尺度 BN 统计不同,混采污染原型空间。"""
        m = emb_maps[si]                                       # [1,512,H,W]
        Hf, Wf = m.shape[-2:]
        e_map = F.normalize(m[0], dim=0).permute(1, 2, 0)      # [H,W,512] 单位化
        out = []
        for j, (x1, y1, x2, y2) in enumerate(np.asarray(boxes_xyxy, np.float32)):
            if masks is not None:
                cell = cv2_resize_bool(masks[j], Wf, Hf)
            else:
                cell = np.zeros((Hf, Wf), bool)
                gx1, gy1 = int(x1 / STRIDES[si]), int(y1 / STRIDES[si])
                gx2 = min(int(np.ceil(x2 / STRIDES[si])), Wf)
                gy2 = min(int(np.ceil(y2 / STRIDES[si])), Hf)
                cell[gy1:gy2, gx1:gx2] = True
            if not cell.any():                                 # 太小:退化为中心点
                cy = int(np.clip((y1 + y2) / 2 / STRIDES[si], 0, Hf - 1))
                cx = int(np.clip((x1 + x2) / 2 / STRIDES[si], 0, Wf - 1))
                cell[cy, cx] = True
            v = e_map[torch.from_numpy(cell)].mean(0)
            out.append(F.normalize(v, dim=0, p=2))
        return (torch.stack(out) if out
                else torch.zeros(0, m.shape[1], device=m.device))

    # ── ① pf 通路 ────────────────────────────────────────────────────
    @torch.no_grad()
    def propose(self, img_hwc: np.ndarray, conf=0.05):
        """pf 端到端提案。→ (boxes [N,4], confs [N], masks [N,384,640] bool|None)。"""
        r = self.pf.predict(img_hwc, imgsz=(384, 640), conf=conf,
                            max_det=self.max_det, verbose=False, device=self.dev)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), None
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        masks = None
        if r.masks is not None:
            m = r.masks.data.cpu().numpy() > 0.5               # [N,h,w](可能非全尺寸)
            if m.shape[-2:] != (384, 640):
                import cv2
                m = np.stack([cv2.resize(x.astype(np.uint8), (640, 384),
                                         interpolation=cv2.INTER_NEAREST) > 0
                              for x in m])
            masks = m
        return boxes, confs, masks

    # ── 融合:token 流 ────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, img_hwc: np.ndarray, bank: torch.Tensor, conf=0.05):
        """单帧 → (tokens [N, 6+C], masks)。列 = [cx,cy,w,h(归一), conf, area, cos×C]。"""
        boxes, confs, masks = self.propose(img_hwc, conf)
        emb_maps = self.embed(img_hwc)
        e = self.proposal_embed(emb_maps, boxes)               # [N,512]
        cos = (e @ bank.T).cpu().numpy() if len(e) else np.zeros((0, bank.shape[0]))
        geo = np.zeros((len(boxes), 6), np.float32)
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            geo[j] = [(x1 + x2) / 2 / 640, (y1 + y2) / 2 / 384,
                      (x2 - x1) / 640, (y2 - y1) / 384, confs[j],
                      (x2 - x1) * (y2 - y1) / (640 * 384)]
        return np.concatenate([geo, cos.astype(np.float32)], 1), masks
