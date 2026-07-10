# 当前系统实际在用的论文与外部成果清单

> 口径:只登记**当前 GRPO 快塔线**(Omni 慢塔 + 像素快塔[DINO 前端接线中] + Haiku 判官 + CraftGround)
> 真正吃到的外部成果。已退役的世界模型 / 预训练 / BC 蒸馏线不在此列(见文末)。
> 纪律:每条给 `文件:行号` 证据;arXiv 编号、权重 ID 一律从仓库原文抄录,查不到写 `待补`,不猜。

## 一、部件 × 成果 对照总表

| 部件 | 借用的成果 | 编号 / 权重 ID | 我们用了它的什么 | 明确没用它的什么 | 仓库内证据 |
|---|---|---|---|---|---|
| 慢塔底座 | Nemotron-3-Nano-Omni | arXiv:2604.24954;HF `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-{BF16,FP8,NVFP4}`(NVIDIA Open Model Agreement) | 读一帧游戏画面 → 文本子目标 + 目标像素;NVFP4 本地 vLLM 加载 | 不解码长思维链驱动 30Hz 动作;不做像素直控(零样本失败) | `docs/architectures/nemotron-3-nano-omni-architecture.md:10-14`;`train/craftground/grpo_pixel.py:62,97-98,116-124` |
| 慢塔量化格式 | Omni 官方 NVFP4/FP8 分层量化 | 同上权重(NVFP4 变体) | 专家张量 NVFP4 + 注意力/共享专家 FP8,5090 单卡自托管 | Mamba 本体不 4bit 化(官方原样) | `docs/architectures/nemotron-3-nano-omni-architecture.md:44-50`;`knowledge/conclusion_omni_nvfp4_5090.md` |
| 慢塔提示范式 | Lumine | arXiv:2511.08892(Qwen2-VL-7B 底座玩原神) | 语言原生动作串 + 显式推理 + 多帧历史 + 像素指点配方逐条移植 | 未用其原神微调权重;我们是零样本,未 SFT | `knowledge/conclusion_omni_pixel_control.md:12`;`tests/probe_omni_minecraft_lumine.py:4`;`docs/activity_log.md:349` |
| 慢塔像素指点坐标约定 | Omni 自身被训练过的表示 | 1000×1000 归一化像素坐标 | `aim` 点用 0..1000 归一坐标(实测标定出的原生表示) | 角度/度数标定不归它(符号量级不可靠) | `knowledge/conclusion_omni_pixel_control.md:82`;`train/craftground/grpo_pixel.py:76-82` |
| 句向量编码器 | Sentence-Transformers MiniLM | `sentence-transformers/all-MiniLM-L6-v2`(384d,冻结) | 把慢塔文本子目标编码成 goal 向量喂快塔 FiLM 条件 | 不微调、不做检索 | `train/craftground/grpo_pixel.py:326-331` |
| 判官 | Anthropic Claude Haiku | `claude -p --model haiku`(CLI) | 组内并行 rollout 从好到差排名 → 名次取负 → 组内 z 归一当相对优势 | 不产生动作、不做感知;只给序 | `train/craftground/grpo_pixel.py:187`;`tests/probe_judge_io_haiku.py:2-16` |
| 快塔视觉前端(2026-07-10 用户拍板方向,接线中) | DINOv3(Meta,自监督 ViT) | HF repo 见 `net/backbone.py` `_HF_REPOS`(dinov3 ViT-S/16,gated;dinov2 开放降级);arXiv **待补** | 冻结 patch 网格(保空间结构)作快塔视觉 token + 地图稠密写入;fovea 双尺度裁剪备选 | 不用 CLS 单向量(旧 BC 用法已退役);不微调骨干 | `net/backbone.py:21-43`;`tests/probe_dino_aim.py`;`knowledge/design_bitter_lesson_map_integration.md §7/§8` |
| 快塔相机动作头 | OpenAI VPT(部分在用) | GitHub `openai/Video-Pre-Training`(MIT);vendored `net/vpt_lib/`;arXiv **待补** | 相机 mu-law 11-bin 分箱口径(避开 MSE"恒预测 0"平凡解);20 键契约 | 不用它的 BC 预训练 / 软 KL 蒸馏 / 逆动力学(distill_vpt 退役) | `train/minecraft/vpt_action.py:13,19-30`;`net/vpt_lib/NOTICE:1-8`;`net/pixel_tower.py:18-20` |
| 快塔卷积干 | IMPALA-CNN(风格引用) | arXiv **待补**;仓库注为"OpenAI VPT / snu-mllab Achievement-Distillation 的 IMPALA-CNN" | 从零手写"IMPALA 风格"小卷积干(不 import 现成实现,不载预训练) | 不用其残差深塔 / 预训练权重 | `net/pixel_tower.py:77`;`blocks/impala.py:8` |
| 训练算法 | GRPO(组内相对优势策略梯度) | arXiv **待补**(仓库无论文引用) | 判官排序 → 组内 z 归一优势 → REINFORCE(`loss=adv·(CE+BCE)`) | 当前实现是 REINFORCE 变体,未加 importance-ratio/clip/KL(待补全成完整 GRPO) | `train/craftground/grpo_pixel.py:1-23,195-200,272-298`;`train/fovea_twotower/grpo_harness.py:52-55` |
| 方法论立场 | Sutton《The Bitter Lesson》 | 仓库原文写 "Sutton 2019"(随笔,非 arXiv) | 不为单个游戏打人工感知补丁;裁决退役词表/凸包GT/手标分割头 | 反对的是人工领域先验,非大规模预训练通用表征 | `net/pixel_tower.py:3`;`train/craftground/grpo_pixel.py:6-9`;`docs/next_session.md:146-147` |

## 二、"部分在用"两条的边界说明

### VPT —— 动作表示在用,BC 路线退役
- **在用**:相机 mu-law 离散分箱头的口径(`CAMERA_BINS=11` / `CAMERA_MU`),理由是 MSE 回归下"恒预测 0"是平凡解,mu-law 分箱把 0 变成众多类之一;VPT 原版同样用 mu-law 离散相机。20 个二值键的动作契约同源。见 `train/minecraft/vpt_action.py:10-13,29-30`。
- **未用**:`net/vpt_lib/` 虽是 OpenAI VPT 策略网络的原样 vendored 副本,但其用途(`distill_vpt.py` 的 teacher 软 KL 蒸馏)属退役的 BC 预训练路线。见 `net/vpt_lib/NOTICE:5-6`。

### YOLOE —— 整线废弃(2026-07-10 用户拍板)
- 2026-07-09 曾裁决路线 2(类别无关提案 token);**2026-07-10 用户按苦涩的教训再推一步
  直接拍板 DINO,YOLOE 整线废弃**:YOLOE = 通用特征 + 人工监督的 objectness 头,删
  n_cls 只删了一半人工先验,物体性先验仍在(地形/洞/坡不成"物体",结构性缺席);
  DINO patch 网格语义等价或更强、建图明确更强(稠密地面可写图)。
- 代码已删(`yolo_unified/yolo_parse/seg_head/token_stream/wood/tower` 及全部依赖方,
  git 历史 commit `f6e997b` 及之前可查);原"DINO vs YOLOE 探针门控"作废,探针降级为
  DINO 单臂瞄准可学性验证(`tests/probe_dino_aim.py`)。
- 覆盖率实测(pf 90.8% vs 4 类词表 1.3%)作为**词表塌缩的证据**仍有效,登记表引用不动。

## 三、疑似在用但无书面出处(待人工确认)

- **IMPALA-CNN 架构本体**(Espeholt 等):`net/pixel_tower.py:77` 只写"IMPALA 风格",`blocks/impala.py:8` 只注"照搬 OpenAI VPT / snu-mllab Achievement-Distillation 的实现",**仓库无 arXiv 编号与原作者**。待人工确认是否登记原始论文。
- **MobileCLIP 文本塔**:`docs/architectures/fovea-hypothesis-verification-2026-07-08.md:76` 提到 YOLOE 自带 MobileCLIP 文本塔(`mobileclip_blt.ts`),但路线 2 去掉了词表/文本点名,当前是否在用不明确,**无 arXiv**。待确认。
- **REINFORCE / 策略梯度**(Williams;Sutton & Barto):作为 GRPO 更新的底层,反复出现于代码注释,但**无任何书面引用**。属基础方法,通常不单列。

## 四、曾经在用、现已退役(只列名字)

- **Dreamer4 世界模型** —— 世界模型主线,已弃。
- **DreamerV3(Crafter)** —— `net/dreamerv3`,arXiv:2301.04104,清白重建但不在快塔线。
- **VPT BC 预训练 / 软 KL 蒸馏** —— `distill_vpt.py`,动作表示保留、蒸馏退役。
- **gaming500 预训练** —— 带动作游戏视频自监督,退役。
- **PPO + Achievement-Distillation** —— `net/ppo_ad`,退役。
- **冻结 DINOv3 骨干 BC** —— `net/bc/policy.py`,按苦涩的教训退役。
- **YOLOE 感知先验路线 1**(词表 + conv 分割头 + 类向量 bank)—— `g1_conv_head` / `g1_vectors` / `token_stream.py`,退役。
- **YOLOE 路线 2**(类别无关提案 token)—— 2026-07-10 用户拍板废弃(objectness 仍是人工先验),整线代码已删,方向改 DINO patch 网格。
- **Nemotron-TwoTower(扩散语言建模)** —— arXiv:2606.26493,不在快塔线。
- **HunyuanVideo-1.5 / HunyuanGameCraft** —— arXiv:2506.17201 / 2511.23429,仅架构调研。
