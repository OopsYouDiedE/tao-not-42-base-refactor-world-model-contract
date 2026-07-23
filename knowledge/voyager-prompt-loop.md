2026-07-23 | Voyager 提示词与迭代循环:结构化状态→自然语言对话的翻译机制(arXiv 2305.16291 + MineDojo/Voyager 开源代码) | B/1(方法·官方代码实读)·B/4(性能自报利益相关) | 更新于2026-07-23;失效:上游仓库大改 prompt 布局或换 LLM 接口后重读
---
Voyager——用 LLM 玩 Minecraft 的开放式 agent(NVIDIA GEAR + Caltech 等)。对本项目的价值**不在它玩游戏,在它把"结构化环境状态/命令翻译成 LLM 对话"的工程范式**——这正是 obs→action 数据与 VLA 策略侧要复用的东西。以下均从官方开源代码第一手读出,非论文转述。

【四 agent 闭环】不是单循环,是四个独立 LLM agent 分工(`voyager/voyager.py`):
- **CurriculumAgent**(出题·自动课程):基于当前状态提下一个**单**任务,格式硬约束 `Mine/Craft/Smelt/Kill/Cook/Equip [数量] [对象]` 单短语。**禁止需视觉确认的任务**(建造/挖洞/种植/交易)——因为 critic 只能读状态字段判成败,看不到画面。
- **ActionAgent**(执行·迭代提示):写 Mineflayer JS 代码。
- **CriticAgent**(评判·自我验证):强制 JSON `{"reasoning","success","critique"}`,只看最终状态(尤其 inventory)判成败。
- **SkillManager**(技能库):给每段成功代码生成一句话 NL 描述存向量库,下轮按 context 检索 top-k 塞回 ActionAgent 的 system message。
外层 `learn()`:出题→rollout→成功则存技能→更新完成/失败清单。内层 `step()`:LLM 写码→`env.step(code)` 跑→critic 判→重组消息带报错+critique→成功或重试用尽(默认 4 次)则退出。

【★核心:状态→对话的翻译(本项目最该抄的部分)】`action.py:render_human_message` 把结构化状态**逐字段拼成自解释文本块**:
```
Code from the last round: <上轮代码>
Execution error: <运行报错>
Chat log: <游戏内输出>
Biome/Time/Nearby blocks/Nearby entities/Health/Hunger/Position/Equipment
Inventory (2/36): {'oak_log': 2}
Task: Mine 3 wood logs
Context: ... / Critique: <critic 上轮批评>
```
三个可直接迁移的设计要点:
1. **反馈全部回灌进下一轮 prompt**:报错、聊天日志、critic 批评都拼进 human message,形成"环境反馈→代码修正"闭环。不重开对话,把新观察当作用户下一句话。
2. **观察自解释**:字段名旁边直接写判读提示(`Health: Higher than 15 means healthy`、`Inventory (xx/36)` 暗示槽位上限),模型无需额外背景就能读懂。
3. **只保留两条消息滑动窗口**(`self.messages = [system_message, human_message]`):**不累积对话历史**,每轮把最新状态重渲成一条 human message,上下文靠"上轮代码+报错+critique"几个字段承载。对控 token / 防上下文漂移关键。

【system vs human 分工】system(`action_template.txt`)静态:设角色+注入可用 helper 程序(`{programs}` 占位,填技能库检索到的代码)+编码规范(复用 `mineBlock` 而非 `bot.dig`、禁死循环、禁 `bot.on` 事件监听、`maxDistance=32` 不作弊)。human 每轮动态生成。输出用 ```javascript``` 包裹便于正则提取(`action.py` 用 `re.compile(r"```(?:javascript|js)(.*?)```")`),babel 解析取最后一个 async 函数、强校验签名必须 `(bot)` 单参。

【对本项目教训】①"状态→自解释文本块 + 反馈回灌 + 两消息滑窗"可直接做成 obs 渲染器原型,喂给动作 token 策略;与本项目 [[qwen3vl-action-token-probe]] 的"图文交错布局"结论互补——Voyager 是纯文本状态,本项目要把观测帧图像插进去。②critic 只读状态字段判成败=省了视觉判官,但代价是"禁止一切需视觉确认的任务";本项目有真实观测帧,可判更丰富的任务,不必受此限。③Voyager 动作是**LLM 现写 JS 高层原语**(mineBlock 等),非本项目的低层键+相机 token——两者动作抽象层级不同,它的"技能库复用代码"范式适合高层规划层,不适合底层动作 codec。④与 [[lumine-agent-recipe]] 对比:Lumine 是纯 BC 学底层键序,Voyager 是 LLM 现场编程+自我验证,两条路线正交。
---
来源:
- 论文 arXiv 2305.16291(TMLR 2024;作者 NVIDIA GEAR / Caltech / 斯坦福 / UT Austin / ASU,Jim Fan 团队,该领域有往绩→信源 B)。
- 官方开源 MIT 许可仓库 https://github.com/MineDojo/Voyager(本条所有机制均从下列文件第一手读出):
  - `voyager/voyager.py`:四 agent 闭环、`learn/rollout/step`、两消息滑窗、重试上限。
  - `voyager/agents/action.py`:`render_human_message`(状态→对话逐字段翻译)、`render_system_message`(注入 programs)、`process_ai_message`(babel 解析 JS)。
  - `voyager/prompts/{action_template,critic,curriculum,skill,action_response_format}.txt`:各 agent 系统提示词原文。
- 内容评级理由:机制/提示词=官方开源代码逐行实读,第一手原始出处+逻辑自洽→1;论文性能宣称(比 AutoGPT 探索快 3.3× 等)自报无本项目复现+利益相关→4。
