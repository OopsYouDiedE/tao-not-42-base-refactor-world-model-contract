"""Minecraft student (轻量级 IMPALA CNN)。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinecraftStudent(nn.Module):
    def __init__(self, img_shape=(3,128,128), hidsize=256, num_actions=27, impala_width=1):
        super().__init__()
        c = int(16 * impala_width)
        self.conv = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *img_shape)
            cnn_out = self.conv(dummy).numel()
        self.fc = nn.Linear(cnn_out, hidsize)
        self.pi_head = nn.Linear(hidsize, num_actions)

    def forward(self, x):
        x = self.conv(x)
        hidden = F.relu(self.fc(x.flatten(1)))
        logits = self.pi_head(hidden)
        return logits, hidden
