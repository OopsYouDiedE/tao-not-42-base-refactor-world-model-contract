"""集成冒烟：两步式 MinecraftWorldModel(离散词表自回归 + Cross-Attn 重建) 前向 + 反向 + 动作分词器联合训练。

离线 CPU 运行，绕开网络/GPU，只验证管线接线，无 NaN 梯度。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.world_model import MinecraftWorldModel
from net.config import ModelConfig, DynamicsConfig, HeadsConfig
from net.action_model import ActionTokenizer, ActionExecutor
from train.minecraft.losses import vocab_pred_loss, z_recon_loss
from train.minecraft.eval import evaluate

ACT_DIM = 22


class MockDINOv2(nn.Module):
    """随机冻结卷积，模拟 DINOv2 输出 patch token。"""
    def __init__(self, d=64):
        super().__init__()
        self.embed_dim = d
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=8), nn.ReLU(),
            nn.Conv2d(64, d, kernel_size=4, stride=4), nn.ReLU(),
        )

    def forward(self, x):
        feat = self.net(x)                                  # [B, d, h, w]
        B, d, h, w = feat.shape
        return feat.view(B, d, h * w).transpose(1, 2)       # [B, M, d]


def _tiny_model():
    cfg = ModelConfig(d=64, K=2, J=2, act_dim=ACT_DIM, max_skip=3,
                      dynamics=DynamicsConfig(num_layers=2, nhead=4))
    return MinecraftWorldModel(cfg, backbone=MockDINOv2(64))


def test_world_model_forward_backward():
    B, device = 2, "cpu"
    model = _tiny_model().to(device).train()

    img = torch.rand(B, 3, 64, 64, device=device)
    z_ref = model.encode_obs(img)                       # [B, M, d]
    M = z_ref.shape[1]
    assert z_ref.shape == (B, M, model.d)

    h = torch.randn(B, 1, model.d, device=device)
    a_hist = torch.zeros(B, model.J, ACT_DIM, device=device)
    a_cur = torch.zeros(B, model.S, ACT_DIM, device=device)
    dt = torch.full((B,), float(model.S), device=device)
    t_vec = torch.zeros(B, device=device)
    target_token_id = torch.randint(0, 512, (B,), device=device)

    out = model(z_ref, h, a_hist, a_cur, dt, t_vec, target_token_id=target_token_id)
    assert out["logits"].shape == (B, 512)
    assert out["z_recon"].shape == (B, M, model.d)
    assert out["h_next"].shape == (B, 1, model.d)

    # 损失反向传播
    l_vocab = vocab_pred_loss(out["logits"], target_token_id)
    l_recon, _ = z_recon_loss(out["z_recon"], z_ref)
    loss = l_vocab + l_recon
    loss.backward()

    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in model.parameters())
    assert not has_nan, "梯度出现 NaN"


def test_action_tokenizer_and_executor():
    """测试离散动作分词器 ActionTokenizer 和执行器 ActionExecutor。"""
    B, device = 2, "cpu"
    action_tok = ActionTokenizer(act_dim=ACT_DIM, hidden_dim=64, latent_dim=64, n_embed=512).to(device).train()
    action_exec = ActionExecutor(act_dim=ACT_DIM, latent_dim=64, state_dim=64, hidden_dim=64, max_skip=3).to(device).train()

    a_cur = torch.rand(B, 3, ACT_DIM, device=device)
    valid_mask = torch.ones(B, 3, device=device)
    dt = torch.full((B,), 3, device=device)

    # 1. 压缩编码为离散 Token
    z_q, indices, tok_loss = action_tok(a_cur, valid_mask=valid_mask)
    assert z_q.shape == (B, 64)
    assert indices.shape == (B,)
    assert tok_loss.shape == ()

    # 2. 执行器还原动作
    z_ref = torch.rand(B, 8, 64, device=device)  # 模拟环境特征
    a_recon = action_exec(z_q, z_ref, dt)
    assert a_recon.shape == (B, 3, ACT_DIM)

    # 3. 联合反向
    loss = tok_loss + F.mse_loss(a_recon, a_cur)
    loss.backward()

    has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                  for p in list(action_tok.parameters()) + list(action_exec.parameters()))
    assert not has_nan, "ActionModel 梯度出现 NaN"


def test_evaluate_smoke():
    """evaluate() 接口端到端冒烟测试。"""
    B, T, device = 2, 4, "cpu"
    model = _tiny_model().to(device)
    action_tok = ActionTokenizer(act_dim=ACT_DIM, hidden_dim=64, latent_dim=64, n_embed=512).to(device)
    
    batch = {
        "img": torch.rand(B, T, 3, 64, 64, device=device),
        "act_seq": torch.rand(B, T, 3, ACT_DIM, device=device),
        "dt": torch.randint(1, 4, (B, T), device=device).float(),
        "t_vec": torch.arange(T, device=device).float().unsqueeze(0).expand(B, T).clone(),
    }
    
    out = evaluate(model, action_tok, [batch], device, amp_dev="cpu", use_amp=False)
    assert "loss" in out
    assert "vocab_acc" in out
    assert "recon_ratio" in out


if __name__ == "__main__":
    test_world_model_forward_backward()
    test_action_tokenizer_and_executor()
    test_evaluate_smoke()
    print("ok")
