"""LLM 指导层离线冒烟(CPU,小尺寸)(tests/integration/)。

验证嵌合插槽的形状自洽与梯度隔离:
    - 目标条件化 critic:use_goal 下 behavior.loss / planner.act 全链前向+反向;
    - SemanticRewardHead:作 reward_fn 注入想象回报,goal=None 时通道置零(防火墙档);
    - GuidanceBus:发布/推进/陈旧降级(mock 编码器按 AGENTS §2 依赖注入)。
"""
import hashlib

import torch
import torch.nn.functional as F

from net.dreamerv3 import build_dreamerv3
from net.dreamerv3.planner import Planner
from net.guidance import GuidanceConfig, SemanticRewardHead, build_semantic_reward
from utils.guidance_bus import GuidanceBus


GD = 16   # 冒烟用文本嵌入维(生产 MiniLM = 384)


def mock_encode(texts):
    """确定性 mock 句向量:md5 种子单位随机向量,[B, GD] fp32(依赖注入,非生产代码)。"""
    out = []
    for s in texts:
        seed = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(GD, generator=g)
        out.append(v / v.norm())
    return torch.stack(out)


def _build_agent():
    return build_dreamerv3(
        device="cpu", num_actions=7, obs_shape=(3, 64, 64),
        dyn_deter=32, dyn_stoch=4, dyn_discrete=4, dyn_hidden=32,
        units=32, mlp_layers=1, enc_depths=(8, 16, 32, 64),
        dec_depths=(64, 32, 16, 8), horizon=4,
        use_goal=True, goal_text_dim=GD)


def test_goal_value_semantic_reward_cpu():
    torch.manual_seed(0)
    agent = _build_agent()
    head = build_semantic_reward(
        agent.world_model.feat_dim, goal_text_dim=GD, units=32, mlp_layers=1)

    B, T = 2, 5
    obs = torch.rand(B, T, 3, 64, 64)
    action = F.one_hot(torch.randint(0, 7, (B, T)), 7).float()
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    goal = mock_encode(["chop tree"] * (B * T)).reshape(B, T, GD)

    wm_loss, post, _ = agent.world_model.loss(
        obs, action, torch.randn(B, T), torch.ones(B, T), is_first)
    post_sg = {k: v.detach() for k, v in post.items()}

    a_loss, v_loss, m = agent.behavior.loss(
        post_sg, agent.world_model, goal=goal, reward_fn=head.reward)
    assert torch.isfinite(a_loss) and torch.isfinite(v_loss)
    (a_loss + v_loss).backward()
    # 语义奖励在 no_grad 下混入 ⇒ 头本身不接收想象梯度(蒸馏训练在 train/)
    assert all(p.grad is None for p in head.parameters())
    # goal=None ⇒ 语义通道置零(北极星防火墙档)
    feat = torch.randn(3, agent.world_model.feat_dim)
    assert head.reward(feat, None).abs().sum() == 0

    planner = Planner(agent, n_candidates=4, horizon=3, discount=0.99,
                      goal_align_coef=1.0, use_goal=True)
    latent = agent.world_model.dynamics.initial(B, "cpu")
    idx, onehot = planner.act(latent, goal[:, 0])
    assert idx.shape == (B,) and onehot.shape == (B, 7)


def test_guidance_bus():
    bus = GuidanceBus(mock_encode, static_plan=("explore",), stale_after_s=0.0)
    g = bus.read()
    assert g.source == "static" and g.subgoal == "explore"
    assert g.goal_vec.shape == (GD,)

    bus.publish_plan(["chop tree", "make table"])
    # stale_after_s=0 ⇒ 读取即判陈旧,应降级回静态计划
    assert bus.read().source == "static"

    bus2 = GuidanceBus(mock_encode, static_plan=("explore",), stale_after_s=60.0)
    bus2.publish_plan(["chop tree", "make table"])
    assert bus2.read().subgoal == "chop tree"
    assert bus2.advance() and bus2.read().subgoal == "make table"
    assert not bus2.advance()          # 计划末项,等待重编译
    assert bus2.staleness() < 60.0


if __name__ == "__main__":
    test_goal_value_semantic_reward_cpu()
    test_guidance_bus()
    print("OK")
