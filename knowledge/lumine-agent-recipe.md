2026-07-21 | Lumine 动作格式/训练配方/算力(arXiv 2511.08892v1) | F/2(方法事实)·F/4利益相关(性能宣称) | 更新于2026-07-21;失效:论文出新版或放权重后重评
---
Lumine——玩《原神》并零样本迁移《鸣潮》《崩铁》的通用游戏agent。"开放配方"=方法公开,**未确认放权重/代码**。

【骨干+数据】Qwen2-VL-7B-Base(论文只标"7B",精确参数量NOT STATED)。预训练1731h(2424h过滤)+指令200h+推理15h/15K traces+分类器标注165h。720p输入。

【动作格式·核心】感知5Hz(200ms/帧一次前向),电机30Hz:每200ms窗吐6个33ms chunk,每chunk 0–4键。时长靠run-length(键连续出现=保持按下不重按;缺席=自动松开),**无时长值/无按下-松开事件token**→规避"3秒对2秒错"脆弱性的机制。鼠标ΔX/ΔY整数(−1000,1000),滚轮ΔZ[−5,5],delta平滑铺200ms执行。格式`<|action_start|> ΔX ΔY ΔZ ; K ;...; K <|action_end|>`,分号落地即流式发送。**纯BC,无IDM无RL**(自动标注用Qwen2-VL-2B分类器)。

【算力·Table 3(H100)】三阶段:预训练→指令→推理。batch 128(推理64),Batch Packing Length 32768。**non-history合计~6,400 H100-h**(预训练5376+指令960+推理64)。**history(20帧滑窗)~20,736 H100-h**(预训练19008+指令1664+推理64)。⚠️开视觉历史把预训练干到3.5×(5376→19008)——注意力窗扛历史=每步重算整段图像token。部署4×H20(非H100)/TP=4/W8A8·SmoothQuant/首chunk 113.9ms(带思考234ms);decode 3.1ms/tok、prefill 52ms、vision encoder 39ms;无草稿模型投机解码(靠固定分隔符)。

【⚠️注意:这些是Lumine的微调账,非从头预训练】6.4k/20.7k H100-h是在Qwen2-VL-7B-Base之上的BC,底座预训练是阿里沉没成本。**勿当"从头训能打游戏模型"的价签**。

【NOT STATED(勿当已知)】精确参数量、总训练token/帧数、推理吞吐(tok/s)、权重是否已放。

【对本项目教训】①"开历史贵3.5×"是"选递归态可躲此税"的证据,非"已选Mamba"(骨干未定)。②run-length chunk可抄格式,但片长改Minecraft 50ms/tick(200ms=4片,非6片33ms)。③无音频、无IDM——数据不足1731h量级仍需自建IDM。

---
来源:arXiv 2511.08892v1(NVIDIA无关,作者机构无往绩→信源F)。
- 摘要页 https://arxiv.org/abs/2511.08892:频率、游戏列表。
- 全文 https://arxiv.org/html/2511.08892v1:Table3算力/硬件、Table4延迟、batch/epoch、数据小时、动作格式、run-length原句"Keys that appear in consecutive chunks retain their key-down state and are not pressed again"、RL不实际的表述。
- 内容评级理由:方法/算力取自论文自表,逻辑自洽但无独立证实→2;性能宣称(完成5h主线/零样本迁移)自报无第三方复现+利益相关→4;权重未放故无法独立验证。
