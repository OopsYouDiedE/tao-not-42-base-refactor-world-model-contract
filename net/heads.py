"""Minecraft 世界模型的解码头。

提供：
    ActionVocabHead      — 预测最有可能的离散动作 Token (第一步的词表推断)
"""
import torch
import torch.nn as nn


class ActionVocabHead(nn.Module):
    """用于推断最可能离散动作 Token 词表的分类头。"""
    def __init__(self, d, vocab_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Sequential(
            nn.Linear(d, 128),
            nn.SiLU(),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        """输入特征 x: [B, L, d] 或 [B, d]。

        返回 Token 的分类 Logits: [B, L, vocab_size] 或 [B, vocab_size]。
        """
        return self.head(x)

