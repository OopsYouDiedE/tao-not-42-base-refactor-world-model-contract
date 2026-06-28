"""Fake teacher (随机策略)。"""
import torch
import torch.nn as nn

class FakeTeacher(nn.Module):
    def __init__(self, img_shape=(3,128,128), hidsize=256, num_actions=27):
        super().__init__()
        self.hidsize = hidsize
        self.num_actions = num_actions

    def forward(self, x):
        B = x.shape[0]
        logits = torch.randn(B, self.num_actions, device=x.device)
        hidden = torch.randn(B, self.hidsize, device=x.device)
        return logits, hidden
