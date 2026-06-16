"""vendored DreamerV3(net/dreamer/)的加载与运行冒烟:CPU、离线、tiny 尺寸。

验证"原封不动照抄 + 物理拆进 net/blocks 后仍可完整加载并跑通":
  1) blocks 算子:symexp∘symlog 还原 / OneHotDist 直通采样 / DiscDist two-hot log_prob 形状。
  2) build_dreamer 构造完整 DreamerV3(WorldModel + ImagBehavior),参数量 > 0。
  3) WorldModel._train 一步:编码→RSSM.observe→各头 log_prob→KL→优化器反向,指标有限。
  4) ImagBehavior._train 一步:想象 rollout→λ-return→actor/critic 损失→反向,指标有限。
无 pytest 依赖(纯 assert + __main__,与本仓其余测试一致);不触网络/GPU/数据集。
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, _ROOT)

from blocks.distributions import symlog, symexp, OneHotDist, DiscDist
from net.dreamer import build_dreamer

A = 6          # 动作维
IMG = (16, 16, 3)


def _tiny():
    """构造 CPU 上的 tiny DreamerV3(16×16 图像,小隐藏维,短想象地平线)。"""
    return build_dreamer(
        num_actions=A,
        obs_shapes={"image": IMG},
        device="cpu",
        dyn_stoch=4, dyn_discrete=4, dyn_deter=32, dyn_hidden=32, units=32,
        imag_horizon=3,
        encoder={"cnn_depth": 8, "minres": 4},
        decoder={"cnn_depth": 8, "minres": 4},
    )


def _data(B=4, T=6):
    """随机一段 (B,T) 轨迹:图像 [0,255]、动作连续、reward 连续、首帧 is_first=1。"""
    g = np.random.RandomState(0)
    is_first = np.zeros((B, T), np.float32)
    is_first[:, 0] = 1.0
    return {
        "image": (g.rand(B, T, *IMG) * 255).astype(np.float32),
        "action": g.randn(B, T, A).astype(np.float32),
        "reward": g.randn(B, T).astype(np.float32),
        "is_first": is_first,
        "is_terminal": np.zeros((B, T), np.float32),
    }


def test_blocks_distributions():
    x = torch.linspace(-50, 50, 101)
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-3), "symexp∘symlog 未还原"

    # OneHotDist:采样是 one-hot 且带直通梯度(可反传到 logits)
    logits = torch.randn(3, 5, requires_grad=True)
    s = OneHotDist(logits, unimix_ratio=0.01).sample()
    assert s.shape == (3, 5)
    assert torch.allclose(s.sum(-1), torch.ones(3)), "采样非 one-hot 分布"
    s.sum().backward()
    assert logits.grad is not None, "直通梯度未回传到 logits"

    # DiscDist:two-hot symexp 头的 log_prob 形状 = 去掉 bucket 维
    dd_logits = torch.randn(2, 7, 255)
    lp = DiscDist(dd_logits, device="cpu").log_prob(torch.randn(2, 7))
    assert lp.shape == (2, 7), lp.shape


def test_build_loads():
    agent = _tiny()
    n_wm = sum(p.numel() for p in agent.world_model.parameters())
    n_bh = sum(p.numel() for p in agent.behavior.parameters())
    assert n_wm > 0 and n_bh > 0, "DreamerV3 未成功构造参数"
    # discrete 隐变量 feat = stoch*discrete + deter
    assert agent.world_model.embed_size > 0


def test_world_model_train_step():
    agent = _tiny()
    wm = agent.world_model
    post, context, metrics = wm._train(_data())
    # post 为 RSSM 后验状态(已 detach),含 discrete logit/stoch + deter
    assert set(post) >= {"stoch", "deter", "logit"}
    for k in ("model_loss", "image_loss", "reward_loss", "cont_loss", "kl"):
        assert k in metrics and np.isfinite(metrics[k]), f"{k} 缺失或非有限"


def test_behavior_train_step():
    agent = _tiny()
    wm, behavior = agent.world_model, agent.behavior
    post, _, _ = wm._train(_data())

    # 行为学习的奖励目标:对想象状态读世界模型的奖励头(与原仓 dreamer.py 一致)
    reward = lambda f, s, a: wm.heads["reward"](wm.dynamics.get_feat(s)).mode()
    imag_feat, imag_state, imag_action, weights, metrics = behavior._train(post, reward)

    assert torch.isfinite(imag_feat).all(), "想象特征出现非有限值"
    assert "actor_loss" in metrics and np.isfinite(metrics["actor_loss"])
    assert "value_loss" in metrics and np.isfinite(metrics["value_loss"])


if __name__ == "__main__":
    test_blocks_distributions()
    test_build_loads()
    test_world_model_train_step()
    test_behavior_train_step()
    print("ok")
