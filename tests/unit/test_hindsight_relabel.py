# -*- coding: utf-8 -*-
"""hindsight relabel 契约单测:事件倒推窗口 / GUI 命名 / aim 坐标 / goal 386 维契约
(与 grpo_pixel.SlowTower 逐字节同式)/ FiLM 梯度真流 / 数据集 goal 装配。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.minecraft.hindsight_relabel import (GOAL_DIM, MODEL_ID,  # noqa: E402
                                               build_goal, frame_facts,
                                               label_frames)
from train.minecraft.vpt_dataset import assemble_goal  # noqa: E402


def _raw(stats=None, gui=False, mouse=None, keys=()):
    return json.dumps({"stats": stats or {}, "isGuiOpen": gui,
                       "mouse": mouse or {"x": 0, "y": 0, "dx": 0, "dy": 0},
                       "keyboard": {"keys": list(keys)}})


def _clip(events_at):
    """构造 n 帧原始行,events_at = {帧号: {stat 键: 增量}}(累计计数器自动累加)。"""
    n = max(events_at) + 3
    cum, lines = {}, []
    for t in range(n):
        for k, dv in (events_at.get(t) or {}).items():
            cum[k] = cum.get(k, 0) + dv
        lines.append(_raw(stats=dict(cum)))
    return lines


def test_window_and_nearest_event():
    """帧 f 的标签 = f 后 window 帧内最近事件;窗口外无标签;帧 0 差分基线抑制。"""
    lines = _clip({10: {"minecraft.mine_block:minecraft.oak_log": 1},
                   14: {"minecraft.pickup:minecraft.oak_log": 1}})
    labels = label_frames(frame_facts(lines), window=5)
    assert labels[0] is None and labels[5] is None          # 窗口外
    assert labels[6][0] == "mine oak log" and labels[10][0] == "mine oak log"
    assert labels[11][0] == "collect oak log"               # 更近的下一事件
    assert labels[14][0] == "collect oak log"
    assert labels[15] is None
    # 累计计数器在帧 0 就非零也不得算事件(会话续录基线不可信)
    lines2 = [_raw(stats={"minecraft.mine_block:minecraft.stone": 7})] * 4
    assert all(lb is None for lb in label_frames(frame_facts(lines2), window=5))


def test_same_frame_priority():
    """同帧多事件按固定类别优先级:craft > mine > kill > pickup > drop > use。"""
    lines = _clip({5: {"minecraft.use_item:minecraft.torch": 1,
                       "minecraft.pickup:minecraft.stone": 1,
                       "minecraft.craft_item:minecraft.stick": 2}})
    labels = label_frames(frame_facts(lines), window=3)
    assert labels[4][0] == "craft stick"


def test_gui_naming_and_close():
    """开 GUI 命名取开沿 [0,+1] 帧 custom 增量;无增量=E 背包;关沿同名 close。"""
    lines = []
    cum = {"minecraft.custom:minecraft.play_one_minute": 0}   # 计步器类不产事件
    for t in range(12):
        cum["minecraft.custom:minecraft.play_one_minute"] += 1
        if t == 4:
            cum["minecraft.custom:minecraft.interact_with_crafting_table"] = \
                cum.get("minecraft.custom:minecraft.interact_with_crafting_table", 0) + 1
        gui = 3 <= t <= 7
        lines.append(_raw(stats=dict(cum), gui=gui,
                          mouse={"x": 640.0, "y": 360.0, "dx": 0, "dy": 0}))
    labels = label_frames(frame_facts(lines), window=2)
    assert labels[2][0] == "open crafting table" and labels[3][0] == "open crafting table"
    assert labels[7][0] == "close crafting table" and labels[8][0] == "close crafting table"
    # 无 interact 增量的 GUI = E 背包
    lines2 = [_raw(gui=(2 <= t <= 4), mouse={"x": 640, "y": 360, "dx": 0, "dy": 0},
                   stats={"minecraft.custom:minecraft.jump": t}) for t in range(8)]
    labels2 = label_frames(frame_facts(lines2), window=2)
    assert labels2[2][0] == "open inventory" and labels2[5][0] == "close inventory"


def test_aim_coordinates():
    """非 GUI 事件 aim=(500,500) 画面中心;GUI 事件 aim=光标 1280×720→0..1000。"""
    lines = _clip({4: {"minecraft.mine_block:minecraft.dirt": 1}})
    labels = label_frames(frame_facts(lines), window=3)
    assert labels[4][1] == (500, 500)
    # GUI 内 craft 事件:aim 取事件帧光标
    lines2 = []
    cum = {}
    for t in range(8):
        if t == 5:
            cum["minecraft.craft_item:minecraft.stick"] = 1
        lines2.append(_raw(stats=dict(cum), gui=t >= 2,
                           mouse={"x": 320.0, "y": 180.0, "dx": 0, "dy": 0}))
    labels2 = label_frames(frame_facts(lines2), window=3)
    assert labels2[4][0] == "craft stick"
    assert labels2[4][1] == (250, 250)                       # 320/1280, 180/720


def test_goal_contract_matches_slowtower():
    """goal = MiniLM 向量 ⊕ aim/1000,386 维,与 grpo_pixel.SlowTower:219 逐字节同式;
    模型 id 与 grpo_pixel 同串(词向量空间必须一致)。"""
    v = torch.randn(384); v = v / v.norm()
    aim = (730, 415)
    g = build_goal(v, aim)
    ref = torch.cat([v, torch.tensor([aim[0] / 1000.0, aim[1] / 1000.0])])  # SlowTower 式
    assert g.shape == (GOAL_DIM,) and torch.equal(g, ref)
    cfg = PixelTowerConfig(goal_dim=384 + 2)
    assert cfg.goal_dim == GOAL_DIM
    src = Path("train/craftground/grpo_pixel.py").read_text(encoding="utf-8")
    assert MODEL_ID.split("/")[-1] in src


def test_assemble_goal_zero_when_unlabeled():
    """无标签帧(id=-1)goal 全零(与既有零 goal 行为兼容);有标签帧 = 向量⊕aim/1000。"""
    mat = torch.eye(3, 384)
    ids = torch.tensor([-1, 2, 0])
    aim = torch.tensor([[0., 0.], [500., 500.], [1000., 250.]])
    g = assemble_goal(ids, aim, mat)
    assert g.shape == (3, 386) and torch.equal(g[0], torch.zeros(386))
    assert torch.equal(g[1, :384], mat[2]) and torch.allclose(g[1, 384:],
                                                              torch.tensor([0.5, 0.5]))
    assert torch.allclose(g[2, 384:], torch.tensor([1.0, 0.25]))


def test_film_gradient_flows():
    """FiLM 路径梯度真流:goal 非零时 goal_q/goal_bias 有梯度,且换 goal 换输出。"""
    torch.manual_seed(0)
    cfg = PixelTowerConfig(img_hw=(32, 32), goal_dim=386, d=32, layers=1, heads=2)
    tower = build_pixel_tower(cfg)
    img = torch.rand(2, 1, 3 * cfg.frame_stack, 32, 32)
    prev = torch.zeros(2, 1, cfg.n_mouse + cfg.n_keys)
    goal = torch.zeros(2, 386); goal[0, 7] = 1.0; goal[0, 384:] = 0.5
    cam_l, key_l = tower(img, goal, prev)
    (cam_l.sum() + key_l.sum()).backward()
    assert tower.goal_q.weight.grad.abs().sum() > 0
    assert tower.goal_bias.weight.grad.abs().sum() > 0
    with torch.no_grad():
        goal2 = goal.clone(); goal2[0, 7] = -1.0
        cam_b, _ = tower(img, goal2, prev)
    assert not torch.allclose(cam_l, cam_b)                  # goal 内容改变 ⇒ 行为通道响应
