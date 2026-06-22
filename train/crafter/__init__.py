"""Crafter 训练域 (train/crafter/)。

覆盖内容:
    env.py         — VecCrafterEnv 向量化环境封装。
    rollout.py     — RolloutBuffer PPO 轨迹缓冲区。
    ad_buffer.py   — AchievementBuffer Achievement Distillation 示范缓冲区。
    ppo_loss.py    — ppo_loss 损失计算。
    train_ppo_ad.py— PPO+AD 训练主程序 CLI。
    dreamer_buffer.py — SequenceReplay 序列回放缓冲区(DreamerV3 用)。
    train_dreamerv3.py — DreamerV3 训练主程序 CLI。
"""
