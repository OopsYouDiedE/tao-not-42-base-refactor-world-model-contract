# CraftGround keyboard_and_mouse_only 后端

## 身份

| 项目 | 值 |
| --- | --- |
| 项目后端名称 | `keyboard_and_mouse_only` |
| 文本协议 | `standard-input-action/v1` |
| CraftGround 上游枚举 | `ActionSpaceVersion.V2_MINERL_HUMAN` |
| 支持设备 | `KeyboardMouse` |
| 暂不支持设备 | `Gamepad`、`Touch` |

`V2_MINERL_HUMAN` 只作为调用 CraftGround 第三方 API 时使用的上游标识。项目文档、指标和适配器使用 `keyboard_and_mouse_only`，避免将上游动作空间版本误认为标准输入动作协议版本。

## 当前映射

`CraftGroundKeyboardMouseAdapter` 把转译器产生的单个 `ActionTick` 转换为 CraftGround `no_op_v2()` 完整动作字典。每个协议 tick 对应一次 `environment.step()`，环境负责实际耗时。

| 协议输入 | CraftGround V2 字段或行为 |
| --- | --- |
| `W` / `S` / `A` / `D` | `forward` / `back` / `left` / `right` |
| `Space` / `Shift` / `Ctrl` | `jump` / `sneak` / `sprint` |
| `MouseLeft` / `MouseRight` | `attack` / `use` |
| `Q` / `E` | `drop` / `inventory` |
| `1`–`9` | `hotbar.1`–`hotbar.9` |
| `MouseMove x y` | `camera_yaw=x*0.15`、`camera_pitch=y*0.15` |
| `Scroll delta` | 根据已记录快捷栏位置循环选择 `hotbar.1`–`hotbar.9` |
| `NoOp` | 未覆盖任何按键字段的完整 `no_op_v2()` 字典 |

适配器每 tick 都从完整 no-op 字典开始，因此上一 tick 缺席的按键会释放。同一 tick 的多个快捷栏数字键，以及滚轮与数字键并用会被拒绝。当前执行闭环仅支持 `KeyboardMouse`；`Gamepad` 与 `Touch` 仍不接入 CraftGround。
