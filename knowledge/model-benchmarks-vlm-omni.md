2026-07-21 | 候选模型性能:Qwen3-VL / Nemotron Nano 12B v2 VL / Nemotron-3-Nano-Omni-30B-A3B / Gemma 4(26B-A4B·31B) | 厂商自报B/2利益相关无独立证实;gemma-4与Qwen3-VL-235B有LMArena独立数据 | 更新于2026-07-21;失效:独立榜收录更多型号后重评
---
候选骨干选型的性能对照。**全部厂商自报、利益相关**;截至查证日,独立第三方评测(OpenCompass OpenVLM、LMArena Vision、Artificial Analysis、Video-MME官方)**均未收录本项目实际考虑的型号**(仅收了不用的 Qwen3-VL-235B),故厂商数**无法交叉证实,内容级封顶2**。

## ⚠️两家可直接对比的只有2项(其余各报各套件,不可比)
- **MathVista_MINI**:Qwen3-VL-30B-A3B 81.9(think)/80.1(inst) ≈ Omni 82.8 → 打平。
- **Video-MME**:Qwen3-VL-30B-A3B **74.5(think)/77.3(inst)** > Omni **72.2** → Qwen视频更强(⚠️协议存疑:Omni卡未标 w/ 还是 w/o 字幕;Qwen是w/o sub)。
- 结论:选型别信"30B比12.6B视频强"——目前**无数据支持**,且Omni 30B的Video-MME反低于同级Qwen 30B-A3B。

## 纠错:"72.2 vs 74.5"不是跨厂商对比
两数都是NVIDIA自报、不同benchmark:72.2=Omni的Video-MME;74.52=Omni的Daily Omni(视频+音频)。非Nemotron-vs-Qwen。

## Qwen3-VL(Alibaba,利益相关)
**⚠️版本现状(2026-07-21查证HF API):Qwen3-VL仍是最新生成式VL,无Qwen3.5-VL/Qwen4-VL**(API精确搜索空数组、猜测仓库401=不存在非拒访)。Qwen文本旗舰已到3.5(2026-02)/3.6(2026-04)但均无视觉;LMArena上"qwen3.5"是纯文本跑视觉任务、非VL版。VL线落后文本一代。发布时间线:235B-A22B(2025-09-22)→30B-A3B(09-30)→4B/8B(10-11)→32B(10-19)→全系GGUF(10-31)→VL-Embedding/Reranker 2B/8B(2026-01-07,检索用非生成)。**最新生成式VL即30B-A3B(09-30);8B(10-11)有官方FP8/GGUF,5090部署省事。**

尺寸:2B/4B/8B/32B稠密 + 30B-A3B(31B总/~3B激活)/235B-A22B(236B总/22B激活),各有Instruct/Thinking+FP8。上下文原生256K可扩1M。数字=think/inst,源arXiv 2511.21631v2 Table2-4,Video-MME取w/o sub行。

| bench | 235B | 32B | 30B-A3B | 8B | 4B |
|---|---|---|---|---|---|
| MMMU | 80.6/78.7 | 78.1/76.0 | 76.0/74.2 | 74.1/69.6 | 70.8/67.4 |
| MathVista | 85.8/84.9 | 85.9/83.8 | 81.9/80.1 | 81.4/77.2 | 79.5/73.7 |
| DocVQA | 96.5/97.1 | 96.1/96.9 | 95.5/95.0 | 95.3/96.1 | 94.2/95.3 |
| OCRBench | 875/920 | 855/895 | 839/903 | 819/896 | 808/881 |
| Video-MME(w/o) | 79.0/79.2 | 77.3/76.6 | 74.5/77.3 | 71.8/71.4 | 68.9/69.3 |
| MVBench | 75.2/76.5 | 73.2/72.8 | 72.0/72.3 | 69.0/68.7 | 70.8/68.9 |
| MLVU | 83.8/84.3 | 82.3/82.1 | 78.9/81.3 | 75.1/78.1 | 75.7/75.3 |
| LVBench | 63.6/67.7 | 62.6/63.8 | 59.2/62.5 | 58.0/56.2 | 55.0/55.2 |

NOT REPORTED:LongVideoBench、TempCompass。

## Nemotron Nano 12B v2 VL(NVIDIA,利益相关)
⚠️**独立HF卡GATED(401)**,取不到自家卡;下列数来自**Omni卡里的"Nano VL V2"对比列**(即同一上游NVIDIA转报,非独立第二来源)。用的是与Qwen完全不同的套件,无法与上表对比。
CVBench2D 78.3 | OCRBenchV2(EN) 54.8 | OSWorld 11.1 | CharXiv-Reasoning 41.3 | MMLongBench-Doc 38 | MathVista 75.5 | OCR_Reasoning 33.9。
NOT REPORTED:Video-MME、MVBench、MMMU、DocVQA、上下文、精确参数量("12B"仅来自命名)。

## Nemotron 版本现状(2026-07-21查证HF API)
- **当前世代=Nemotron 3**(Nano/Super/Ultra,collection `nvidia-nemotron-v3`);**无比3更新的世代**。⚠️HF上`Nemotron-4-*`(340B)是**2024旧线**,NVIDIA重排编号,4比3更老,勿混。
- **Omni线:`nemotron-3-nano-omni-30b-a3b-reasoning`(BF16 2026-04-20、FP8/NVFP4 04-24)是最新omni**,之后仅Audex(2026-07-06音频)、Nemotron-3-Embed(07-14文本),非全模态。
- **VL线:12B-v2-VL(2025-10-21)是最新主流instruct VLM,但非最新vision模型**——更新的有`Nemotron-Labs-Diffusion-VLM-8B`(2026-05-08,扩散VLM、image-text-to-text**无视频**、Labs研究模型)、VL-embedding(`llama-nemotron-embed-vl-1b-v2` 2026-05-14检索用)。扩散VLM对流式逐帧控制无益(解码非瓶颈、双向打架流式),非候选但呼应"扩散语言模型"话题。

## Nemotron-3-Nano-Omni-30B-A3B-Reasoning(NVIDIA,利益相关)
架构Mamba2-Transformer混合MoE;骨干Nemotron3 Nano LLM(30B-A3B)+C-RADIO v4-H视觉编码器+Parakeet语音编码器。**总31B,激活~3B/token,上下文256k**。输入视频/音频/图像/文本→输出文本。源:官方HF卡(reasoning模式),arXiv 2604.24954。
CVBench2D 83.95 | OCRBenchV2(EN) 67.04 | OSWorld 47.4 | CharXiv-Reasoning 63.6 | MMLongBench-Doc 57.5 | MathVista 82.8 | OCR_Reasoning 54.14 | **Video-MME 72.2** | World Sense(视+音) 55.4 | Daily Omni(视+音) 74.52 | 语音指令跟随 89.39。
音频/ASR(BF16列):MMAU 74.62 | Tedium-Long WER 3.11 | HF-ASR WER 5.95。VoiceBench命名但无分→NOT REPORTED。
NOT REPORTED:MMMU、DocVQA、ChartQA、AI2D、MMBench、MVBench、MLVU。

## Gemma 4(Google,利益相关;但有独立LMArena数据)
2026-04发布,技术报告arXiv 2607.02770(2026-07-02)。系列E2B/E4B/12B/26B-A4B/31B。**纯decoder-only Transformer,局部滑窗+全局注意力交替(5:1),无Mamba**。Apache 2.0开源。
- **26B-A4B(MoE)**:25.2B总/3.8B激活,256K上下文,**Text+Image(无音频)**。MMLU-Pro 82.6|GPQA-D 82.3|LiveCodeBench-v6 77.1|MMMU-Pro 73.8|MATH-Vision 82.4(厂商)。**LMArena Vision rank39/Elo1240、Text rank70(独立)**。
- **31B(稠密)**:30.7B(含视觉编码器~550M共34B),256K,**Text+Image(无音频)**。MMLU-Pro 85.2|GPQA-D 84.3|LCB-v6 80.0|Codeforces 2150|AIME2026 89.2|MMMU-Pro 76.9|MATH-Vision 85.6(厂商)。**LMArena Vision rank26/Elo1255、Text rank50(独立)**。
- ⚠️**视频最弱**:26b/31b原生不支持视频,仅抽帧当图像(**上限60秒、1fps**),**未报任何视频benchmark**。音频仅E2B/E4B/12B有。
- ⚠️**纠错**:用户印象"31b几乎第一"与live榜不符;Google自报也仅称"开源稠密模型第一"(#43,2026-06-19快照),非总榜第一。可能看了"筛选开源"视图。
- 5090适配NOT CONFIRMED(官方无VRAM表;有int4/w4a16/q4_0 QAT档,30.7B稠密4bit约16GB权重+KV,推测可塞但未见官方claim)。
- **独门优势**:三家候选中**唯一有独立验证**(LMArena真人盲评),性能数可达内容级2且有独立交叉;Qwen/Nemotron只能停在利益相关自报。

## 独立评测覆盖(交叉证实用)
截至2026-07-21:**Qwen3-VL 30B-A3B/8B、Nano 12B v2 VL、Omni 30B-A3B在所有独立榜均NOT LISTED**;**gemma-4-31b/26b-a4b已被LMArena收录(见上,真独立)**;Nemotron系全缺席(用户亦确认LMArena无NVIDIA mamba模型)。唯一有独立数据的是Qwen3-VL-235B-A22B(不用):LMArena人类盲评Elo instruct 1215(#53)/thinking 1189(#67)=真独立(可重跑);Artificial Analysis智能指数14(估计值、偏文本)。OpenCompass OpenVLM数据冻结在2025-09-17,早于全部目标模型发布→缺席=快照过期非拒收。无ACCESS DENIED;Papers with Code已停站。

---
来源:
- Qwen:arXiv 2511.21631v2(Qwen3-VL Technical Report)Table2-4,PDF逐页读取(HF卡benchmark是JPG图无法提取)。Alibaba利益相关。
- Nemotron Omni:官方HF卡 https://huggingface.co/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning。NVIDIA利益相关。
- Nano 12B v2 VL:自家卡 https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL **GATED 401**(确认为门控非临时错误);数据仅来自Omni卡对比列。
- 独立榜:OpenVLM原始JSON(opencompass.openxlab.space,数据戳20250917)、arena.ai/leaderboard/vision、artificialanalysis.ai、video-mme.github.io。
- Gemma 4:arXiv 2607.02770(Gemma Team,利益相关)+ HF卡 google/gemma-4-26B-A4B-it、gemma-4-31B-it-qat-w4a16-ct + ai.google.dev/gemma/docs(2026-04-02)+ arena.ai/leaderboard[/vision](独立)。四源日期/数字互相咬合。
- 评级理由:厂商自报有确凿出处+逻辑自洽但利益相关且无独立证实→2;Nano 12B v2数为同一上游转报,更弱;Qwen3-VL-235B与gemma-4的LMArena Elo独立可重跑→那两条有独立交叉;gemma-4是本项目候选中唯一有独立验证者。
