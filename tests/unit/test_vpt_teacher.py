# -*- coding: utf-8 -*-
"""VPT 教师翻译层单测(CPU,不需要权重文件)。

锚定内容:
  1. 相机下推表:教师 bin 中心角度经 encode(deg_to_bins)∘decode(undiscretize)
     的组合是确定、对称、单调的,且质量守恒;
  2. 键位置换是同名双射(零手写映射的前提);
  3. teacher_to_v2 在解析构造的分布上给出精确边缘概率(one-hot 联合类 → 0/1 键概率;
     camera 元动作关 → 相机质量全在中心 bin);
  4. remap_cam 概率质量守恒。
"""
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.vpt_lib.action_mapping import CameraHierarchicalMapping
from net.vpt_lib.actions import Buttons, CameraQuantizer
from train.craftground.action_contract import CAM_BINS, V2_KEYS, bins_to_deg, deg_to_bins
from train.minecraft.vpt_teacher import (TEACHER_CAM_KWARGS, TEACHER_CAM_NULL,
                                         TEACHER_KEY_TO_V2, N_BUTTON_COMBOS,
                                         remap_cam, teacher_bin_pushforward,
                                         teacher_to_v2, _MAPPING)


def test_pushforward_table():
    """下推表:每行恰一个 1(教师 ±10° 被我们 ±18° 覆盖)、中心对中心、对称、单调。"""
    p = teacher_bin_pushforward()
    assert p.shape == (11, CAM_BINS)
    assert (p.sum(1) == 1).all()                       # 质量守恒(行随机矩阵)
    m = p.argmax(1).numpy()
    assert m[5] == CAM_BINS // 2                       # 教师零 bin → 我们零 bin
    assert (m[10 - np.arange(11)] == (CAM_BINS - 1) - m).all()   # 符号对称
    assert (np.diff(m) >= 0).all()                     # 单调不减
    # encode∘decode 一致:表 = deg_to_bins(教师 bin 中心角度),逐点复算锚定
    q = CameraQuantizer(**TEACHER_CAM_KWARGS)
    assert (deg_to_bins(q.undiscretize(np.arange(11))) == m).all()
    # 我们侧自身 encode∘decode 恒等(bin 中心不动点)
    b = np.arange(CAM_BINS)
    assert (deg_to_bins(bins_to_deg(b)) == b).all()


def test_key_permutation_bijection():
    """键位置换:同名双射,置换后名字逐一相等。"""
    assert sorted(V2_KEYS) == sorted(Buttons.ALL)
    assert len(set(TEACHER_KEY_TO_V2)) == 20
    for v2_i, t_i in enumerate(TEACHER_KEY_TO_V2):
        assert Buttons.ALL[t_i] == V2_KEYS[v2_i]


def _onehot_logits(idx_buttons: int, idx_cam: int):
    """构造 one-hot 联合分布的 log-prob(数值上用大负数代替 -inf)。"""
    lb = torch.full((1, N_BUTTON_COMBOS), -1e4)
    lb[0, idx_buttons] = 0.0
    lc = torch.full((1, 121), -1e4)
    lc[0, idx_cam] = 0.0
    return {"buttons": lb, "camera": lc}


def test_translate_onehot_meta_on():
    """forward+attack+camera 开的联合类 → 精确 0/1 键概率;相机边缘取对应 bin。"""
    comb = ("none", "forward", "none", "none", "none", "none", "attack", "none", "camera")
    bi = _MAPPING.BUTTONS_COMBINATION_TO_IDX[comb]
    # camera 联合 idx:pitch bin 2, yaw bin 8(camera_combinations = product(x=pitch, y=yaw))
    ci = _MAPPING.camera_combination_to_idx[("camera_x2", "camera_y8")]
    p_keys, cam_t, p_on = teacher_to_v2(_onehot_logits(bi, ci))
    expect = {k: 0.0 for k in V2_KEYS}; expect["forward"] = 1.0; expect["attack"] = 1.0
    for i, k in enumerate(V2_KEYS):
        assert abs(float(p_keys[0, i]) - expect[k]) < 1e-6, k
    assert abs(float(p_on[0]) - 1.0) < 1e-6
    # 轴序:cam_t[...,0,:]=dx(yaw bin 8),cam_t[...,1,:]=dy(pitch bin 2)
    assert abs(float(cam_t[0, 0, 8]) - 1.0) < 1e-6
    assert abs(float(cam_t[0, 1, 2]) - 1.0) < 1e-6


def test_translate_meta_off_and_inventory():
    """camera 元动作关 → 相机质量全并入中心 bin;inventory 特殊类 → 只有 inventory 键。"""
    all_none = tuple("none" for _ in range(9))
    bi = _MAPPING.BUTTONS_COMBINATION_TO_IDX[all_none]
    ci = _MAPPING.camera_combination_to_idx[("camera_x0", "camera_y0")]  # 应被 meta off 覆盖
    p_keys, cam_t, p_on = teacher_to_v2(_onehot_logits(bi, ci))
    assert float(p_keys.abs().sum()) < 1e-5
    assert float(p_on[0]) < 1e-6
    assert abs(float(cam_t[0, 0, TEACHER_CAM_NULL]) - 1.0) < 1e-6
    assert abs(float(cam_t[0, 1, TEACHER_CAM_NULL]) - 1.0) < 1e-6
    bi_inv = _MAPPING.BUTTONS_COMBINATION_TO_IDX["inventory"]
    p_keys, _, _ = teacher_to_v2(_onehot_logits(bi_inv, ci))
    for i, k in enumerate(V2_KEYS):
        assert abs(float(p_keys[0, i]) - (1.0 if k == "inventory" else 0.0)) < 1e-6, k


def test_translate_uniform_mass_conservation():
    """均匀联合分布:键概率=各键在 8641 类中的出现频率;相机边缘和恒 1;remap 守恒。"""
    lb = torch.zeros(3, N_BUTTON_COMBOS) - float(np.log(N_BUTTON_COMBOS))
    lc = torch.zeros(3, 121) - float(np.log(121.0))
    p_keys, cam_t, p_on = teacher_to_v2({"buttons": lb, "camera": lc})
    freq = _MAPPING.BUTTON_IDX_TO_FACTORED.mean(0)     # [20] 教师键序
    for v2_i, t_i in enumerate(TEACHER_KEY_TO_V2):
        assert abs(float(p_keys[0, v2_i]) - float(freq[t_i])) < 1e-5
    assert torch.allclose(cam_t.sum(-1), torch.ones(3, 2), atol=1e-5)
    ours = remap_cam(cam_t)
    assert ours.shape == (3, 2, CAM_BINS)
    assert torch.allclose(ours.sum(-1), torch.ones(3, 2), atol=1e-5)
    # 下推不改变期望角度的符号结构:中心质量落在中心
    assert float(ours[0, 0, CAM_BINS // 2]) > float(ours[0, 0, 0])


def test_distill_kl_zero_at_match_and_masked():
    """distill_kl:学生=教师 ⇒ KL≈0;掩码全 False ⇒ 0;教师零质量 bin 被 eps 兜底有界。"""
    from train.craftground.bc_vpt_warmstart import distill_kl
    torch.manual_seed(0)
    n = 64
    cam_l = torch.randn(n, 2, CAM_BINS)
    p_cam = torch.softmax(cam_l, -1)                   # 教师=学生
    key_l = torch.randn(n, 20)
    p_key = torch.sigmoid(key_l)
    on = torch.ones(n, dtype=torch.bool)
    kc, kk = distill_kl(cam_l, key_l, p_key, p_cam, on)
    assert float(kc) < 1e-4 and float(kk) < 1e-3
    kc0, kk0 = distill_kl(cam_l, key_l, p_key, p_cam, torch.zeros(n, dtype=torch.bool))
    assert float(kc0) == 0.0 and float(kk0) == 0.0
    # 教师某 bin 零质量(下推后我们的 0/10 bin 恰是这种):KL 有界且为正
    p_hard = torch.zeros(n, 2, CAM_BINS); p_hard[..., CAM_BINS // 2] = 1.0
    kc2, _ = distill_kl(cam_l, key_l, p_key, p_hard, on)
    assert np.isfinite(float(kc2)) and float(kc2) > 0


def test_teacher_batch_slice_matches_window():
    """_teacher_batch 的 tick 切片与 _window_batch 的监督 tick 同式(ts=[s-1,T-2])。"""
    from train.craftground.bc_vpt_warmstart import _teacher_batch
    b, t_n, s = 2, 9, 4
    tk = torch.arange(b * t_n, dtype=torch.float16).reshape(b, t_n, 1).expand(b, t_n, 20)
    batch = {"tch_keys": tk.contiguous(),
             "tch_cam": torch.zeros(b, t_n, 2, 11, dtype=torch.float16),
             "tch_on": torch.ones(b, t_n, dtype=torch.bool)}
    out_k, out_c, on = _teacher_batch(batch, s, "cpu")
    ts = torch.arange(s - 1, t_n - 1)
    assert out_k.shape == (b * len(ts), 20) and out_c.shape == (b * len(ts), 2, 11)
    expect = tk[:, ts].reshape(-1, 20).float()
    assert torch.equal(out_k, expect) and bool(on.all())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} 通过")
