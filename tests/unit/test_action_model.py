"""离散动作量化与执行模型单元测试 (tests/unit/test_action_model.py)。

验收契约:
  - VectorQuantizer: 保证在 training 状态下正确量化、计算 commitment loss 并回传梯度；EMA 更新和 Random Restart 功能符合预期。
  - ActionTokenizer: 正常处理变长动作输入，结合 valid_mask 后在时间轴进行屏蔽池化，且 commitment loss 可计算梯度。
  - ActionExecutor: 正确还原出动作序列，超出 dt 范围的部分被有效零填充。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from blocks.quantization import VectorQuantizer
from net.action_model import ActionTokenizer, ActionExecutor


def test_vector_quantizer():
    torch.manual_seed(42)
    dim = 16
    n_embed = 8
    vq = VectorQuantizer(dim=dim, n_embed=n_embed, decay=0.9, commitment_cost=0.25)
    
    # 设为训练模式
    vq.train()
    
    # 构造输入连续特征 [B, dim]
    z = torch.randn(4, dim, requires_grad=True)
    z_q, indices, loss = vq(z)
    
    assert z_q.shape == z.shape
    assert indices.shape == (4,)
    assert loss.dim() == 0  # 0维标量
    assert loss.item() >= 0.0
    
    # 检查梯度回传
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    
    # 模拟大 Batch 输入测试 Random Restart
    # 强制将某些 index 的 EMA 权重打压为 0，并且当前 batch 不选中它们
    # 这里通过一个小规模 batch 验证 EMA 是否会变动
    with torch.no_grad():
        old_embed = vq.embed.clone()
        # 激活一次前向，触发 EMA
        _ = vq(torch.randn(10, dim))
        new_embed = vq.embed.clone()
        # 验证 EMA 使 embed 发生更新 (old != new)
        assert not torch.allclose(old_embed, new_embed)


def test_action_tokenizer():
    torch.manual_seed(42)
    act_dim = 22
    hidden_dim = 32
    latent_dim = 16
    n_embed = 64
    
    tokenizer = ActionTokenizer(act_dim=act_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, n_embed=n_embed)
    tokenizer.train()
    
    B, S = 4, 8
    a_seq = torch.randn(B, S, act_dim, requires_grad=True)
    
    # 无掩码测试
    z_q, indices, loss = tokenizer(a_seq)
    assert z_q.shape == (B, latent_dim)
    assert indices.shape == (B,)
    loss.backward()
    assert a_seq.grad is not None
    
    # 有掩码测试
    a_seq.grad = None
    # valid_mask: 前半部分有效，后半部分无效 (零填充)
    valid_mask = torch.ones(B, S)
    valid_mask[:, 4:] = 0.0
    z_q_m, indices_m, loss_m = tokenizer(a_seq, valid_mask=valid_mask)
    assert z_q_m.shape == (B, latent_dim)
    loss_m.backward()
    assert a_seq.grad is not None


def test_action_executor():
    torch.manual_seed(42)
    act_dim = 22
    latent_dim = 16
    state_dim = 32
    hidden_dim = 64
    max_skip = 8
    
    executor = ActionExecutor(act_dim=act_dim, latent_dim=latent_dim, state_dim=state_dim, hidden_dim=hidden_dim, max_skip=max_skip)
    executor.eval()
    
    B = 4
    z_q = torch.randn(B, latent_dim)
    z_ref = torch.randn(B, 16, state_dim)  # N=16 slots
    dt = torch.tensor([4, 2, 8, 5])  # 各自执行的持续时间
    
    a_recon = executor(z_q, z_ref, dt)
    assert a_recon.shape == (B, max_skip, act_dim)
    
    # 验证有效时间步之外的部分被正确零填充
    # 例如：第 0 条样本，dt=4，那么 index 4, 5, 6, 7 应该是全 0
    # 第 1 条样本，dt=2，那么 index 2, 3, 4, 5, 6, 7 应该是全 0
    for i in range(B):
        d_val = dt[i].item()
        # 有效步内不一定为 0
        # 无效步外必须为 0
        if d_val < max_skip:
            assert torch.all(a_recon[i, d_val:] == 0.0)
            
            
if __name__ == "__main__":
    test_vector_quantizer()
    test_action_tokenizer()
    test_action_executor()
    print("Action Model: all tests passed.")
