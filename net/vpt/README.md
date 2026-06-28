# VPT → CraftGround 蒸馏适配层

## 动作空间映射

**VPT (MineRL):**
- 25 按键: attack/forward/back/jump/left/right/sneak/sprint/use/drop/inventory/hotbar.1-9
- 2D 连续相机: (pitch, yaw) 度数

**CraftGround:**
- 27 离散动作: 基础移动/跳跃/攻击/使用 + 相机(4方向×2幅度) + 12 种组合

**映射规则:**
```python
# 1:1 基础映射
forward→1, back→2, left→3, right→4, jump→5, attack→7, use→9, sneak→11

# 组合动作
forward+jump→6, forward+attack→8, forward+sprint→12

# 相机(阈值:>10°用big,<5°忽略)
pitch>0: look_down(13/17), pitch<0: look_up(14/18)
yaw>0: look_right(15/19), yaw<0: look_left(16/20)

# 无映射 → noop(0)
drop/inventory/hotbar.1-9
```

## 使用

```python
from net.vpt import VPTTeacher, vpt_distill_loss

# 加载 teacher
teacher = VPTTeacher(
    'models/vpt/weights/foundation-model-1x.model',
    'models/vpt/weights/foundation-model-1x.weights',
    target_hidsize=256, target_actions=27
)

# 前向
obs = ...  # (B, 3, H, W)
t_logits, t_hidden = teacher(obs)
s_logits, s_hidden = student(obs, states)

# 蒸馏
loss, metrics = vpt_distill_loss(s_logits, s_hidden, t_logits, t_hidden)
```

## 文件结构

```
net/vpt/
├── __init__.py        # 对外接口
├── adapter.py         # VPTTeacher(投影层)
├── action_mapping.py  # remap_action + 27动作表
├── distill.py         # vpt_distill_loss
└── README.md          # 本文档
```

## 待完成

- [ ] 集成 net/vpt_lib/policy.py 实际加载 VPT weights
- [ ] 处理分辨率差异(VPT 128×128 vs CraftGround 360×640)
- [ ] 离线提取 teacher 特征避免训练时重复推理
