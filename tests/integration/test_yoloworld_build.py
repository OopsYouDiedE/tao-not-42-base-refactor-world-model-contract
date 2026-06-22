"""YOLO-World-Dreamer 离线构建+前向冒烟(CPU,小尺寸)(tests/integration/)。

验证目标条件世界模型 + 256 候选小头 + 双头行为线能构造、形状自洽、前向/反向跑通:
    - 世界模型 loss(含成就头 BCE)前向+反向。
    - 双头行为 loss(cls/plan/align/critic/div/load)前向+反向,且不回梯度到世界模型。
    - 小头点乘选序列的递归 policy 单步。
    - 候选 slot 多样性:全 K 计划不坍缩为同一条。

按 AGENTS §2/§4:CPU 兼容与小尺寸冒烟落在 tests/,生产 net/ 不含降级逻辑。
"""
import torch
import torch.nn.functional as F

from net.yoloworld import build_yoloworld

torch.distributions.Distribution.set_default_validate_args(False)

TINY = dict(
    num_actions=17, obs_shape=(3, 64, 64), n_achievements=22,
    dyn_deter=64, dyn_stoch=8, dyn_discrete=8, dyn_hidden=64,
    units=64, mlp_layers=1, enc_depths=(8, 16, 32, 64), dec_depths=(64, 32, 16, 8),
    n_candidates=32, plan_horizon=4, query_dim=16, head_hidden=48,
    n_rollout=6, n_explore=2)


def _agent_and_E(seed=0):
    torch.manual_seed(seed)
    agent = build_yoloworld(device="cpu", **TINY)
    E = F.normalize(torch.randn(agent.cfg.n_achievements, agent.cfg.task_dim), dim=-1)
    agent.set_ach_embed(E)
    return agent, E


def test_world_model_step_cpu():
    agent, _ = _agent_and_E()
    B, T, U = 2, 5, agent.cfg.n_achievements
    obs = torch.rand(B, T, 3, 64, 64)
    action = F.one_hot(torch.randint(0, 17, (B, T)), 17).float()
    reward, cont = torch.randn(B, T), torch.ones(B, T)
    ach = (torch.rand(B, T, U) > 0.7).float()
    is_first = torch.zeros(B, T); is_first[:, 0] = 1.0

    loss, post, m = agent.world_model.loss(obs, action, reward, cont, ach, is_first)
    assert torch.isfinite(loss)
    loss.backward()
    assert {"image", "reward", "cont", "ach", "kl_dyn", "kl_rep"} <= set(m)


def test_behavior_dual_head_cpu():
    agent, E = _agent_and_E()
    B, T = 2, 5
    obs = torch.rand(B, T, 3, 64, 64)
    action = F.one_hot(torch.randint(0, 17, (B, T)), 17).float()
    reward, cont = torch.randn(B, T), torch.ones(B, T)
    ach = (torch.rand(B, T, agent.cfg.n_achievements) > 0.7).float()
    is_first = torch.zeros(B, T); is_first[:, 0] = 1.0

    _, post, _ = agent.world_model.loss(obs, action, reward, cont, ach, is_first)
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    start = {k: flat(v).detach() for k, v in post.items()}
    N = start["deter"].shape[0]
    task_emb = E[torch.randint(0, agent.cfg.n_achievements, (N,))]

    # 世界模型梯度清零,确认行为线不回梯度到它
    agent.zero_grad(set_to_none=True)
    loss, m = agent.behavior.loss(start, task_emb, agent.proposal, agent.world_model)
    assert torch.isfinite(loss)
    assert {"cls", "actor", "align", "value", "div", "load"} <= set(m)
    loss.backward()
    assert agent.world_model.encoder.layers[0].weight.grad is None  # WM 未收到梯度
    assert agent.proposal.query.grad is not None                    # 小头收到梯度
    agent.behavior.update_slow()


def test_policy_step_and_slot_diversity_cpu():
    agent, E = _agent_and_E()
    B = 3
    task_emb = E[torch.randint(0, agent.cfg.n_achievements, (B,))]
    obs = torch.rand(B, 3, 64, 64)
    idx, onehot, state = agent.policy(obs, None, torch.ones(B), task_emb)
    assert idx.shape == (B,) and onehot.shape == (B, 17)
    idx2, _, _ = agent.policy(obs, state, torch.zeros(B), task_emb, training=False)
    assert idx2.shape == (B,)

    # slot 多样性:全 K 候选首步动作分布不应坍缩为单一(去重的首步 argmax > 1)
    feat = agent.world_model.dynamics.get_feat(state[0])
    plan_logits, _, _ = agent.proposal(feat, task_emb)        # [B, K, H, A]
    first_arg = plan_logits[:, :, 0].argmax(-1)              # [B, K]
    assert first_arg[0].unique().numel() > 1
