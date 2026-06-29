# -*- coding: utf-8 -*-
"""任务文本 → 冻结语义向量(脑内世界的 text token 条件)。

为什么用**冻结**文本编码器而不是可学 embedding:训练集只有 4 个 BASALT 任务,
可学 embedding 等价于 4 个任务 ID 查表——迁移到新指令时没有任何外推可能。
冻结的句向量空间(MiniLM,384 维恰好 = 模型 d)把"语义相近的指令向量相近"
这一先验免费保留下来,prompt-tuning 阶段才有的放矢。

注意期望边界:4 个任务撑不起开放指令跟随,本模块提供的是"任务条件化 + 语义
空间接口";条件是否被模型利用,取决于数据里任务间是否真有行为方差(四任务
混采,见 colab_demo §2)。

 mock 模式:每个唯一字符串一个确定的单位随机向量(md5 种子)——能区分任务、
 无语义外推,供离线冒烟。
"""
import hashlib

import torch
import torch.nn.functional as F


class TaskTextEncoder:
    """冻结句向量编码器。encode(list[str]) -> [B, dim] fp32(CPU 张量,调用方搬运)。

    唯一字符串缓存:任务文本基数极小(每 clip 一句、全库 4 句),逐 batch 编码
    实为查表,开销可忽略。
    """

    DIM = 384            # MiniLM-L6 输出维度(= 模型 d,task_proj 仍保留以解耦)

    def __init__(self, kind="minilm", device="cpu"):
        self.kind = kind
        self.device = device
        self._cache = {}
        self.tok = self.model = None
        if kind == "minilm":
            from transformers import AutoTokenizer, AutoModel
            name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        out = []
        for s in texts:
            s = s or ""
            if s not in self._cache:
                self._cache[s] = self._embed(s)
            out.append(self._cache[s])
        return torch.stack(out)

    def _embed(self, s):
        if self.kind == "minilm":
            t = self.tok(s, return_tensors="pt", truncation=True, max_length=64)
            t = {k: v.to(self.device) for k, v in t.items()}
            h = self.model(**t).last_hidden_state.mean(dim=1).squeeze(0)   # mean pooling
            return F.normalize(h, dim=-1).float().cpu()
        seed = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(self.DIM, generator=g)
        return v / v.norm()
