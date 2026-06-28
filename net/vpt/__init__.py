"""VPT teacher → CraftGround student 蒸馏适配层 (net/vpt/)。

对外接口:
    VPTTeacher       — VPT foundation-model 包装,输出对齐到 CraftGround。
    vpt_distill_loss — KL 散度 + 特征对齐蒸馏损失。
    remap_action     — MineRL 按键+相机 → CraftGround 27动作 id。

VPT(OpenAI Video Pre-Training)在 70K 小时 Minecraft 视频上预训练,本模块将其蒸馏
到 CraftGround PPO+AD 学生模型:动作空间从 MineRL 25键+2D连续相机 映射到 CraftGround
27离散动作,隐藏维度从 1024 投影到 256。
"""
from net.vpt.adapter import VPTTeacher
from net.vpt.distill import vpt_distill_loss
from net.vpt.action_mapping import remap_action, CRAFTGROUND_ACTIONS

__all__ = ["VPTTeacher", "vpt_distill_loss", "remap_action", "CRAFTGROUND_ACTIONS"]
