2026-07-21 | Qwen3-VL-8B 动作 token 零样本探测(本项目自跑实验) | 项目内部实验数据(非外部信源,不适用信源字母评级)·可复现 | 更新于2026-07-21;失效:换模型/数据组/prompt 布局或提高窗口数后重跑
---
**性质说明**:本条目是**项目自跑实验结果**,非外部论文/文档。按知识库 README 定位本属会话记忆库,应项目主人要求收录于此,数值可由 `runs/action_token_probe/report.md` 复现。样本仅 **3 窗口**,窗口间方差远大于效应,绝对值只看方向不当测量。

【实验设置】Qwen3-VL-8B-Instruct **未训练零样本**,HF generate 路径(transformers 5.14.1)。数据 MineStudio 10xx。窗口 209558/236497/36509(按独特性选,max_horizon=20),3窗口×2horizon(5/20)×3条件×10重复。指标=关键动作一致率(只比 移动/转向/姿态/攻击/使用/快捷栏/相机粗方向,抖动不计)。三条件:correct_history(给正确历史动作)、no_history(不给)、random_history(给随机历史)。

【核心结论:交错 prompt 布局的真实增益(同窗口 A/B)】
prompt 从"图像全堆前+文本全塞尾"改为**图文交错**(每帧图像前带时间标签 Frame t-N / Current frame,历史动作 executed: 紧跟其帧)后:

| 条件(全局均值 跨3格式×2horizon) | 堆叠(旧) | 交错(新) | 变化 |
|---|---:|---:|---:|
| correct_history | 0.178 | 0.220 | **+0.042** |
| no_history | 0.000 | 0.000 | **±0.000** |
| random_history | 0.012 | 0.024 | +0.012 |

18 格里约 8 涨 9 平 1 微跌(key_value h5 random −0.018),最大单格 key_value h5 correct_history +0.133。**方向为正、几乎不损,但幅度小且高度依赖窗口**——一次只跑 horizon5 的聚焦对比(另一批窗口)曾报 +0.150、no_history 0→0.33,被完整跑推翻,是小样本假象。

【格式对比(交错布局下 correct_history 均值)】json_line 0.227 ≈ compact_tag 0.217 ≈ key_value 0.217,三者几乎并列,无显著优劣。

【关键发现:no_history 恒为 0】无历史动作时未训练模型完全无法对齐真值,**这是能力问题,prompt 布局改不动**。真正瓶颈是 SFT/LoRA 训练,不是 prompt 工程。

---
【抽样展示 A:典型失败——重复退化】窗口 209558 · compact_tag · horizon20 · correct_history(一致率 0)。模型对全部 20 帧吐同一动作,真值却在变:

| 帧 | 真实动作 | 模型预测 | 命中 |
|---:|---|---|:--:|
| t0 | F sprint use | F sprint | ✗ |
| t2 | F jump sprint use cam(-4,+0) | F sprint | ✗ |
| t8 | F jump sprint use | F sprint | ✗ |
| t19 | F sprint use | F sprint | ✗ |

(全 20 帧预测恒为 `F sprint`;key_value/json_line 同窗口恒为 `F R sprint use`。模型抓到"前进+疾跑"大意,但丢了 use/jump/相机的逐帧变化,退化成单一动作复读。)

【抽样展示 B:部分命中——相机是主要失配点】窗口 36509 · key_value · horizon20 · correct_history(一致率 0.5,本实验最佳单格之一):

| 帧 | 真实动作 | 模型预测 | 命中 |
|---:|---|---|:--:|
| t0 | F jump sprint cam(+0,+1) | F jump sprint | ✓ |
| t3 | F jump sprint cam(-1,+0) | F jump sprint | ✓ |
| t4 | F L jump sprint | F jump sprint | ✗ |
| t8 | F L jump sprint cam(-1,+1) | F jump sprint | ✗ |
| t10 | F jump sprint | F jump sprint | ✓ |
| t13 | F jump sprint cam(-2,+1) | F jump sprint | ✗ |
| t19 | F jump sprint | F jump sprint | ✓ |

(模型稳定输出 `F jump sprint` 主干,命中不含额外键/大相机偏移的帧;失配集中在真值带**左转 L 或明显相机偏移**处——即模型仍不会预测细粒度视角/转向变化。这解释了为何相机粗方向在 field_agreement 里也偏低。)

---
来源:本项目 `runs/action_token_probe/`(report.md / records.jsonl,gitignore 未入库,需本地复现)。生成命令见 `train/minecraft/action_token_probe.py`。交错布局改动见 commit 22881ba(`net/qwen3vl_policy.py`)。
