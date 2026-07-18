# -*- coding: utf-8 -*-
"""OpenAI VPT 策略网络 vendored 副本(教师侧,离线蒸馏用)。

来源:github.com/openai/Video-Pre-Training(MIT License,见本目录 LICENSE),
2026-07-10 重拉；采用边界见 knowledge/README.md §8。改动仅三处:
`from lib.*` → `from net.vpt_lib.*`;actions.py 的 minerl import 改惰性
(仅 item_embed_id_to_name 需要,教师前向不走);policy.py 删无用 `from email import policy`。
网络本体逐行未动。学生侧契约(mu-law 11-bin / 20 键)在
train/craftground/action_contract.py,翻译层在 train/minecraft/vpt_teacher.py。
"""
