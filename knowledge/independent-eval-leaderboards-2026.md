2026-07-21 | 2026现行独立模型评测榜(收开源、类LMArena) | 各榜直取B/F·内容2(存活+日期+独立性已核;具体名次未核) | 更新于2026-07-21;失效:榜停更或跳转后重核
---
"以后要交叉证实开源模型性能,去哪查"的常备参照。判定=真独立(自建评测/众测)vs 转抄厂商数。⚠️具体模型名次经WebFetch摘要层(其cutoff旧、误判2026为虚构),仅榜单清单+存活日期+独立性判定可信,名次需自行点开核。相关:[[model-benchmarks-vlm-omni]]。

## 多模态/VLM(与本项目最相关)
- **LMArena Vision** `arena.ai/leaderboard/vision`(lmarena.ai 301→arena.ai)| 数据2026-07-13,135模型/111万票 | 众测人类盲评,真独立 | 收大量开源VLM | **VLM首选**
- **Artificial Analysis** `artificialanalysis.ai` | 2026-07-20,577模型 | 自建评测(Intelligence Index v4.1),真独立 | **有开源/闭源分栏+Openness Index** | 含视觉/图像/视频/语音
- **Vals AI** `vals.ai` | 2026-07-16 | 全自建in-house,真独立 | **有"Best Open Weight"榜+多模态指数**
- **OpenCompass/OpenVLM** `rank.opencompass.org.cn` | ⚠️本次未能确认(JS站/API超时/HF镜像401);此前发现JSON冻结2025-09-17,未证实未推翻 | 按"可能过期"处理,需浏览器再核

## 纯文本/推理/代码
- **Artificial Analysis** / **Vals AI**(同上,也含文本)
- **LMArena/Arena.ai** `arena.ai/leaderboard` | live 2026 | 众测,LMArena后继,分类最广(Text/Vision/WebDev/Image·Video/Search/Agent/Document)
- **LiveBench** `livebench.ai`(纯JS,经GitHub提交核) | 提交到2026-07-17 | 自跑、抗污染(每月换题/客观标答/无LLM裁判)| 担心刷榜污染时用
- **Aider** `aider.chat/docs/leaderboards` | 2025-11-20(略旧) | 独立可复现代码re-run,带版本/commit/成本

## 别用/慎用
- **HF Open LLM Leaderboard** 已归档(v1 2024-06,v2 2025初)——勿当现行榜。
- **OpenRouter Rankings** 是用量/花费榜,非能力评测(可当独立流行度信号)。
- **Epoch AI / Vellum** 部分转抄厂商自报,交叉证实前逐项核出处。
- **Scale SEAL** 本次未能确认存活(纯JS渲染)。

---
来源:各榜官网直接fetch(2026-07-21),取存活状态/最近数据日期/是否收开源/独立性方法。WebSearch全程424报错,均为直取primary source。摘要层cutoff旧误判2026日期为虚构=其artifact,页面真实,日期按页面采信。
评级理由:各榜作为"自身存在+更新日期+独立性"的信源=单次直取,内容2;榜内具体模型名次未独立核,不入正文当论据。
