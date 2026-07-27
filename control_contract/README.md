# 控制契约：游戏无关、设备无关的纯大模型控制层

本包定义大模型控制电子游戏的唯一动作契约。三条设计约束：

1. **不绑定具体游戏的键位**——角色是语义（`primary` / `interact` / `jump`），物理绑定写在
   每游戏一份的 `BindingProfile` JSON 里。
2. **鼠标键盘与手柄是同一个模型**——任何游戏输入都是**两个抽象摇杆 + 一堆语义按钮**；键鼠
   与手柄的差别只是摇杆参数取值不同，不是两条代码路径。
3. **按大模型的真实能力设计**——一次推理产出一个**时间延展的决策段**（数百毫秒到数秒），
   不是一帧；帧级反应性由编译期固化、运行时零推理成本的**守卫**提供。

## 设备统一模型

| | 位移摇杆 | 瞄准摇杆 | 界面光标 |
|---|---|---|---|
| 手柄 | 连续方向 + 连续力度 | 220°/秒 | 1.2 屏/秒（逐 tick 逼近） |
| Minecraft 键鼠 | 8 向 + 单档（WASD） | 18°/tick | 0.1875 屏/tick（受相机上限约束） |
| 桌面键鼠 | 8 向 + 单档 | 40°/tick | 4.0 屏/tick → **单 tick 跳转** |

"跳转到位置"不是一种独立的设备模型，而是**单 tick 上限 ≥ 屏幕对角线**时逼近逻辑的极限
情形。视角推进与光标推进共用同一个 `_advance_toward`：每 tick 朝目标推进至多
`cap_per_tick`，推不完就如实记账。因此编译器里没有任何 `if 是手柄 / 是鼠标` 的分支。

量化能力用两个整数表达，都是连续可调而非二选一：

- `direction_count`：0 = 任意方向，4 = dpad，8 = WASD，16 = 更细的档位。
- `magnitude_levels`：0 = 连续力度，1 = 开关（键盘），3 = 三档速度。

真正不可调和的只剩**游戏**属性而非设备属性：`menu_cursor=False` 表示这个游戏的界面里根本
没有光标（主机原生 UI），此时 `point` 不可用，改用 `nav_*` + `confirm`。

Minecraft 的 `cursor_cap_per_tick = 0.1875` 不是估算，是从两条实测推出来的：GUI 光标没有
绝对定位通道，只能靠相机增量驱动，所以它继承视角的 **18°/tick** 上限；而光标只走整数像素，
**1 px = 0.15°**（鼠标灵敏度的原生量子）。于是每 tick 最多 18/0.15 = **120 像素**，在 640×360
下等于 0.1875 屏宽 / 0.333 屏高。`cursor_cap_per_tick` 是单标量、按归一化欧氏距离推进，
取两轴中较紧的那个（屏宽向）才不会超卖。像素级标定常量见
`rl_training_environments/craftground/segment_text_codec.py` 顶部。

## 数据流

```text
                       BindingProfile(JSON，每游戏 × 每设备一份)
                                    │
大模型文本 ──segment_codec──▶ Segment ──segment_compiler──▶ DeviceFrame/tick ──adapter──▶ 引擎
                                    │                            ▲
                              guard_monitor ─每 tick 求值─────────┘（命中即截断、抓帧、重推理）
```

- `role_vocabulary.py` — 跨游戏语义角色词表与原语通道名。
- `binding_profile.py` — 唯一允许携带绑定知识的层；`AxisSpec` 是两个摇杆的能力声明，
  `describe_capabilities` 把这些数值翻成给大模型看的自然语言（只列真正支持的原语）。
- `decision_segment.py` — 原语、`Step`、`Guard`、`TailPolicy`、`Segment`、`ControlState`、
  `ExecutionReport`。
- `segment_codec.py` — 文本 ⇄ `Segment`，以及 prompt 侧的格式 / 时序 / 状态说明。
- `segment_compiler.py` — 确定性编译为逐 tick `DeviceFrame`。
- `guard_monitor.py` — 守卫求值与内建像素通道。
- `profiles/` — 内置 profile 数据；新增游戏或设备只加一个 JSON。

## 时间模型

一次推理 = 一个 `Segment`。段内每 tick 的设备输入在编译期就完全确定，运行时不需要推理即可
执行；同时每 tick 求值 `guards`，任一命中就立即截断本段、抓取观测、请求下一轮推理。

| 机制 | 作用 |
|---|---|
| `Step.duration_ms` | 大模型自己决定盲执行多久，单位毫秒（永远看不到 tick） |
| `guards` | 帧级中断条件，纯数值比较，运行时零推理成本 |
| `tail` | 段末到新段生效之间（即推理延迟窗口）继续做什么 |
| `lease_ms` | 死人开关：超时未收到新段即强制释放全部 latch |
| `ExecutionReport` | 回灌"我闭眼这段时间发生了什么"与观测滞后多少毫秒 |

因此：语义决策按秒给出，反应按 tick 兜住，推理延迟期间不停顿也不失控。

## 大模型输出样例

```json
{
  "intent": "walk to the doorway and stop if something blocks me",
  "steps": [
    {"ms": 300, "aim": {"yaw": 25, "pitch": 0}},
    {"ms": 1200, "move": {"dir": 0, "power": 1.0}, "hold": ["sprint"]},
    {"ms": 200, "release": ["sprint"], "press": ["interact"]}
  ],
  "guards": [
    {"channel": "pixel.change", "when": "below", "threshold": 0.01,
     "sustain_ms": 300, "label": "stopped moving, probably blocked"}
  ],
  "tail": "release_move",
  "lease_ms": 1500
}
```

同一段文本喂给三个 profile 会得到不同 tick 数、不同视角分摊、不同位移量化——但大模型写的
东西一个字都不用改。给它看的能力说明也是从 profile 数值现算的，例如手柄会读到
"aim turns at most 220 degrees per second"，键鼠会读到 "move.dir is snapped to the nearest
of 8 directions"，它据此自己判断步长够不够。

## 宽严分工

| 边界 | 策略 |
|---|---|
| `segment_codec.decode_segment` | **宽容**：未知角色丢弃、越界截断、脏文本兜底，永不抛错 |
| `segment_compiler.compile_segment` | **严格**：未知角色与 profile 不支持的原语一律报错 |

前者面向大模型（脏输出必须仍可执行），后者面向程序调用方（写错要立刻暴露）。

## 结构性安全不变量

- 位移用极坐标（方向 + 力度）表达，**结构上不可能**产生前后同按或左右同按。
- 单 tick 视角增量恒不超过 profile 声明的上限，不会被下游编码静默截断。
- 转不完的角度与到不了的光标位置如实记账（`aim_truncation_deg` / `cursor_reached`），
  不静默假装完成。
- 解码永不返回非法段；租约到期永远回到中性态。

## 新增一个游戏

往 `profiles/` 放一个 JSON，声明步频、两个摇杆、光标上限、槽位数与能力别名；再写一个把
`DeviceFrame` 转成该引擎动作格式的 adapter（参考
`rl_training_environments/craftground/control_adapter.py`）。本包与模型端代码都不用改。
