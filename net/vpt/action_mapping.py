"""MineRL → CraftGround 动作映射 (net/vpt/action_mapping.py)。

对外接口:
    CRAFTGROUND_ACTIONS     — 27 动作名称列表。
    remap_action(btn, cam)  — MineRL 动作 → CraftGround 动作 id。
"""
import torch

# CraftGround 27 离散动作(train/craftground/env.py DISCRETE_TO_V2)
CRAFTGROUND_ACTIONS = [
    'noop', 'forward', 'back', 'left', 'right', 'jump', 'forward+jump',
    'attack', 'forward+attack', 'use', 'forward+use', 'sneak', 'forward+sprint',
    'look_down', 'look_up', 'look_right', 'look_left',
    'look_down_big', 'look_up_big', 'look_right_big', 'look_left_big',
    'forward+look_down', 'forward+look_up', 'forward+look_right', 'forward+look_left',
    'attack+look_down', 'attack+look_up'
]


def remap_action(buttons: dict, camera: tuple) -> int:
    """MineRL 动作 → CraftGround 动作 id。

    Args:
        buttons: {'forward': bool, 'attack': bool, ...} MineRL 按键状态
        camera:  (pitch, yaw) float 度数(pitch正=向下,yaw正=向右)

    Returns:
        action_id: 0-26 整数

    映射优先级:组合动作 > 单按键 > 相机 > noop
    """
    fwd = buttons.get('forward', False)
    atk = buttons.get('attack', False)
    use = buttons.get('use', False)
    jmp = buttons.get('jump', False)
    spr = buttons.get('sprint', False)

    pitch, yaw = camera
    has_cam = abs(pitch) > 5 or abs(yaw) > 5

    # 组合动作(优先)
    if fwd and jmp: return 6
    if fwd and atk: return 8
    if fwd and use: return 10
    if fwd and spr: return 12
    if fwd and has_cam:
        if abs(pitch) > abs(yaw):
            return 21 if pitch > 0 else 22
        return 23 if yaw > 0 else 24
    if atk and has_cam:
        return 25 if pitch > 0 else 26

    # 单按键
    if fwd: return 1
    if buttons.get('back'): return 2
    if buttons.get('left'): return 3
    if buttons.get('right'): return 4
    if jmp: return 5
    if atk: return 7
    if use: return 9
    if buttons.get('sneak'): return 11

    # 纯相机
    if has_cam:
        if abs(pitch) > abs(yaw):
            if pitch > 0:
                return 17 if abs(pitch) > 10 else 13
            return 18 if abs(pitch) > 10 else 14
        if yaw > 0:
            return 19 if abs(yaw) > 10 else 15
        return 20 if abs(yaw) > 10 else 16

    return 0
