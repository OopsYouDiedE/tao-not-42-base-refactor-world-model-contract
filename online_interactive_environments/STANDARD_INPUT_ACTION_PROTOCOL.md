# 标准输入动作协议

协议标识：`standard-input-action/v1`

本协议版本描述模型输出的文本结构、设备声明和逐 tick 输入语义。

## 1. 核心定义

记录键盘、鼠标、手柄和移动端的**原始输入格式**，不定义具体业务含义（如 `W` 是否代表前进由各环境的“操纵映射文件”决定）。

---

## 2. 序列结构

### 语法格式

动作序列由设备声明、时间偏移和动作正文组成。`Device` 与 `Tick` 是序列元数据，位于
`<action>` 标签外；`<action>` 标签内只放置可执行的逐 tick 动作：

```text
Device <设备类型>
Tick <时间偏移>
<action><动作正文></action>
```

`parse_action_sequence()` 和 `submit()` 接收且只接收上述单个纯动作序列。包含序列前后解释文本、
多个动作序列或流式分片的模型输出必须交给 `feed()`；`feed()` 保留序列外文本，不把它编译成动作。

* **设备类型**：`KeyboardMouse`（键鼠）、`Gamepad`（手柄）、`Touch`（触摸屏）。*单序列仅支持单设备，切换设备需发新序列。*
* **Tick 偏移**：非负整数。对 `submit()`，序列首个 tick 位于提交时的环境 `current_tick + offset`；对 `feed()`，序列首个 tick 位于完整识别 `Device`、`Tick` 和 `<action>` 头时的环境 `current_tick + offset`。流式序列头一旦识别，锚点保持不变，环境随后推进不重算锚点。首发通常填写 `Tick 0`。
* **动作边界**：`<action>` 和 `</action>` 使用小写并区分大小写。标签内不能出现设备声明或时间偏移。
* **流式提交边界**：流式接收端每收到一个 `;`，立即编译并提交该分号之前的完整 tick；最后一个 tick 在 `</action>` 完整到达时提交。`</action>` 本身是序列结束标记，不要求标签后存在空格、换行或其他字符。尚未出现分号或闭合标签的片段保持缓冲状态，不作为不完整动作执行。
* **序列外文本**：动作序列前后以及多个序列之间的解释文本不属于动作正文。接收端不得把这些文本编译为动作，也不得在处理流式序列时静默丢弃。
* **Tick 分隔**：`<action>` 内用 `;` 分隔连续 tick；同一 tick 内的多个输入用**空格**分隔；无操作的空 tick 写作 `NoOp`。
* **重复**：tick 末尾的 `xN`（`N` 为正整数）表示将该 tick 的完整输入连续执行 `N` 次；动作长度按展开后的 tick 数计算。
* **观察**：`Observe` **独立占据一个分号段**，段内不能出现任何其他输入。它表示在此处进行观测和动作修正：对紧随其后的 tick，在该 tick 输入生效前立即截图，并以该观察异步启动下一轮推理。`Observe` 自身不占用 tick，也不能附加 `xN`（`xN` 只作用于动作段）。`Observe` 之后的旧序列动作是生成当前序列的模型预先提供的延迟填充，不是基于新观察生成的动作。环境在等待新结果时继续执行填充；新结果到达后，从当前 tick 覆盖尚未执行的旧填充，已经执行的填充不回滚。例如 `Observe ; W x8` 表示触发异步重规划，并预先提供最多 8 tick 的 `W` 作为延迟填充。
* **设备路由**：每个已编译 tick 保留所属序列的设备类型。编译器只输出设备和原始输入，不负责环境动作映射。CraftGround 直接消费编译器决策，并在自身边界内完成协议输入到环境动作的映射；不同设备可以共享环境时间线，但 CraftGround 不能把一个设备的输入交给另一个设备的映射器。
* **观察重入**：同一个环境 tick 在确认一次观察后，即使新序列覆盖当前 tick，也不再次产生观察请求。只有提交动作并推进到下一个环境 tick 后，观察资格才重置。

### 2.1 模型生成时钟与环境时钟

模型生成和环境运行使用两套独立时钟：

| 时钟 | 计量内容 | 推进条件 |
| --- | --- | --- |
| 模型生成时钟 | 请求开始、首段内容、首个可执行动作和完整输出耗时 | 由模型接口和编译器接收事件推进 |
| 环境时钟 | `current_tick` 和动作时间线 | 每次环境动作执行完成并提交决策后立即推进 |

模型生成开始、产生分片或结束不会直接推进环境 `current_tick`。环境推进也不会暂停模型生成，二者只在新动作分片进入编译器队列时交换数据。

`ActionSequenceCompiler.feed()` 支持上述流式交换。CraftGround 教师执行器采用更严格的事务边界：先
收齐并校验一轮完整输出的格式、设备、展开后预算和适配器可转换性，再提交并执行。校验失败不会推进
环境。模型请求延迟只写入 `GenerationTelemetry.total_generation_ms`，不调用 `record_wait()`。

环境 tick 没有固定墙钟周期，也不对应帧率或毫秒数。CraftGround 每次 `environment.step()` 完成后立即
拉取、映射并执行下一 tick，不主动休眠或限速，以环境能够完成 step 的最大吞吐速度运行。实际每个
tick 的墙钟耗时由环境计算、进程调度和 IPC 延迟共同决定，因此可能变化。

`Tick` 偏移、等待 tick 数和动作长度都是逻辑 tick 数，不能在没有实测时间戳的情况下换算为毫秒。
需要墙钟性能数据时，由 CraftGround 在每次 step 边界记录开始和结束时间；模型生成耗时继续使用独立
的单调时钟统计。

当前 tick 没有缓存动作时，由编译器的下溢策略决定环境行为：

| 下溢策略 | 编译器决策 | 环境时钟 |
| --- | --- | --- |
| `WAIT` | 返回等待决策，不产生可提交动作 | 停在当前 tick |
| `NOOP` | 返回可提交的 `NoOp` | 提交后继续推进 |
| `REPEAT_LAST` | 返回可提交的上一动作；没有上一动作时返回 `NoOp` | 提交后继续推进 |

只有 `WAIT` 表示“一旦没有新缓存就停下”。其他策略不等待模型生成，环境按照
`step → commit → pull` 循环能跑多快就跑多快。

### 2.1 越界续跑预算

`max_overrun_ticks` 限制的是**队列耗尽之后**允许按下溢策略继续执行的 tick 数，`None` 表示不限。
它不是一次提交总共能执行多少 tick。

队列里仍有排队 tick 而当前 tick 没有动作时，当前 tick 只是队列内部的空隙：`Tick 60` 这类绝对
偏移会在环境 tick 0 与 60 之间留下空隙，环境必须继续推进才能走到下一个有动作的 tick。跨越空隙
按下溢策略执行，但不消耗越界预算，也不使 `overrun_exhausted` 成立——否则 `max_overrun_ticks=0`
会让任何带偏移的提交永远无法执行，而把预算调大又变成“总共只能跑这么多帧”。

因此 `overrun_ticks` 只在队列为空时累加，队列续上后归零；`overrun_exhausted` 要求队列已空且
计数达到预算。

---

## 3. 键鼠协议 (`KeyboardMouse`)

### 3.1 键盘与鼠标按键

* **规则**：出现即代表该 tick 按下，连续按住需逐 tick 重复写出，未出现即释放。
* **键盘名称**：
* 字母：`A`-`Z`（大写） | 数字：`0`-`9`
* 方向键：`Up` / `Down` / `Left` / `Right`
* 修饰键：`Shift` / `Ctrl` / `Alt`（不区分左右，物理键需统一归一化）
* 控制键：`Space` / `Enter` / `Escape` / `Tab` / `Backspace` / `Delete`
* 功能键：`F1`-`F12`


* **鼠标按键**：`MouseLeft` / `MouseRight` / `MouseMiddle` / `MouseButton4` / `MouseButton5`

### 3.2 鼠标移动与滚轮

* **移动**：`MouseMove <x> <y>`（有符号整数，相对位移。`x` 正右负左，`y` 正下负上；`0 0` 可省略）。
* **滚轮**：`Scroll <delta>`（有符号整数）。

---

## 4. 手柄协议 (`Gamepad`)

采用双摇杆 + `A B X Y` 布局，按钮遵循逐 tick 按下/释放规则。

* **按键**：`A` / `B` / `X` / `Y` | `LeftBumper` / `RightBumper` | `LeftStickButton` / `RightStickButton` | `DPadUp` / `DPadDown` / `DPadLeft` / `DPadRight` | `Menu` / `View`
* **摇杆**：`LeftStick <x> <y>` / `RightStick <x> <y>`（`-1.0` 到 `1.0` 浮点数。`x` 正右负左，`y` 正下负上；原点 `0.0 0.0` 可省略）。
* **扳机**：`LeftTrigger <val>` / `RightTrigger <val>`（`0.0` 到 `1.0` 浮点数，`0.0` 为未按，`1.0` 为满按）。

---

## 5. 触摸屏协议 (`Touch`)

坐标使用整数，常见指令如下：

* **点击**：`Tap <x> <y>`
* **长按**：`LongPress <x> <y> <duration_ms>`
* **滑动**：`Swipe <start_x> <start_y> <end_x> <end_y> <duration_ms>`
* **缩放**：`Pinch <center_x> <center_y> <scale>`（`scale` 为正浮点数）

---

## 6. 数据类型与范围汇总

| 指令 | 类型 | 合法范围 |
| --- | --- | --- |
| `Tick` 偏移 / `duration_ms` | 非负整数 | 环境自定义上限 |
| `MouseMove` / `Scroll` / 触摸坐标 | 有符号/无符号整数 | 环境自定义范围 |
| `LeftStick` / `RightStick` | 浮点数 | `[-1.0, 1.0]` |
| `LeftTrigger` / `RightTrigger` | 浮点数 | `[0.0, 1.0]` |
| `Pinch scale` | 正浮点数 | 环境自定义范围 |

### 6.1 v1 校验与宽容规则

解析器按 `Device` 校验指令名称、参数数量、参数类型和固定数值范围。`Device`、`Tick`、
`<action>`、`</action>` 和 `Observe` 是保留控制词，不能作为普通设备输入。

| 问题 | v1 行为 |
| --- | --- |
| 空分号段 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| 未定义指令或设备不匹配 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| 缺少参数或参数类型错误 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| 标签内出现设备/Tick 元数据或嵌套动作标签 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| `Observe` 与动作输入写在同一段内 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| 末尾的 `Observe` 段没有后续动作 | 发出 `RuntimeWarning`，追加一个 `NoOp` tick 承载该观察 |
| 连续多个 `Observe` 段 | 发出 `RuntimeWarning`，只保留一次观察，多余段按 `NoOp` tick 处理 |
| `NoOp` 与其他输入同时出现 | 发出 `RuntimeWarning`，该段按一个 `NoOp` tick 处理 |
| 固定数值范围越界 | 发出 `RuntimeWarning` 并保留原始数值，交由环境映射层决定是否裁剪或拒绝 |

每个由 `;` 或 `</action>` 完整结束的段至少占据一个逻辑 tick。非法段降级为 `NoOp`，不能删除
时间槽，也不能让后续动作提前。`submit()` 与 `feed()` 使用同一套 tick 校验和宽容规则；二者只在
输入边界和序列外文本处理方式上不同。

---

## 7. 操纵映射文件职责

环境需提供独立的映射文件，声明采用的协议版本，用于说明：按键映射、坐标/视角换算、死区、响应曲线、冲突处理和输入上限。环境还需声明 tick 调度模式；v1 的 CraftGround 后端采用无固定周期的最大吞吐模式，并通过实测 step 时间描述性能，不声明固定的单 tick 物理时长。

---

## 8. 综合示例

**键鼠示例**（第 1 帧连发 4 个 tick，第 2 帧延迟 2 个 tick 后发送）：

```text
Device KeyboardMouse
Tick 0
<action>W ; W MouseMove 12 -4 ; MouseLeft Scroll -1 ; NoOp</action>

Device KeyboardMouse
Tick 2
<action>Shift W ; W ; NoOp</action>

```

**观察与重复示例**：

```text
Device KeyboardMouse
Tick 0
<action>W x4 ; Observe ; W x8</action>
```

**手柄示例**：

```text
Device Gamepad
Tick 0
<action>LeftStick 0.0 -1.0 ; LeftStick 0.0 -1.0 RightStick 0.25 0.0 A ; NoOp</action>

```

---

## 9. 兼容性原则

1. 已有指令的名称、参数顺序和数值含义固定。
2. 新增按键或操作为**向后兼容扩展**。
3. 分号只允许在 `<action>` 内用于分隔连续 tick。
4. 变更坐标轴方向、数值范围或 tick 语义需**升级主版本**。
