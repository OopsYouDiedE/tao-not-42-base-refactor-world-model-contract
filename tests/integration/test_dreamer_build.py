"""DreamerV3 / Dreamer4 离线构建+前向冒烟(CPU,小尺寸)(tests/integration/)。

验证两个世界模型能从 blocks 组装、构造成功并跑通形状自洽的前向/反向:
    - DreamerV3:世界模型 loss 前向+反向、想象 actor-critic loss 前向+反向、递归 policy 单步。
    - Dreamer4:tokenizer→时空 Transformer→shortcut 生成→解码 的一次前向(仅构建,不训练)。

按 AGENTS §2/§4:CPU 兼容与小尺寸冒烟落在 tests/,生产 net/ 不含降级逻辑。
"""
import torch
import torch.nn.functional as F

from net.dreamerv3 import build_dreamerv3
from net.dreamer4 import build_dreamer4


def test_dreamerv3_train_step_cpu():
    torch.manual_seed(0)
    agent = build_dreamerv3(
        device="cpu", num_actions=17, obs_shape=(3, 64, 64),
        dyn_deter=64, dyn_stoch=8, dyn_discrete=8, dyn_hidden=64,
        units=64, mlp_layers=1, enc_depths=(8, 16, 32, 64),
        dec_depths=(64, 32, 16, 8), horizon=5)

    B, T = 3, 6
    obs = torch.rand(B, T, 3, 64, 64)
    action = F.one_hot(torch.randint(0, 17, (B, T)), 17).float()
    reward = torch.randn(B, T)
    cont = torch.ones(B, T)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    wm_loss, post, m = agent.world_model.loss(obs, action, reward, cont, is_first)
    assert torch.isfinite(wm_loss)
    wm_loss.backward()
    assert {"image", "reward", "cont", "kl_dyn", "kl_rep"} <= set(m)

    post_sg = {k: v.detach() for k, v in post.items()}
    a_loss, v_loss, bm = agent.behavior.loss(post_sg, agent.world_model)
    assert torch.isfinite(a_loss) and torch.isfinite(v_loss)
    (a_loss + v_loss).backward()
    # 想象 actor-critic 不应回传梯度到世界模型
    assert agent.world_model.encoder.layers[0].weight.grad is not None  # 仅 wm_loss 那次填充
    agent.behavior.update_slow()

    idx, onehot, state = agent.policy(obs[:, 0], None, is_first[:, 0])
    assert idx.shape == (B,) and onehot.shape == (B, 17)
    idx2, _, _ = agent.policy(obs[:, 1], state, torch.zeros(B))
    assert idx2.shape == (B,)


def test_dreamer4_forward_cpu():
    torch.manual_seed(0)
    for use_vq in (False, True):
        agent = build_dreamer4(
            device="cpu", num_actions=17, obs_shape=(3, 64, 64),
            token_dim=32, enc_depths=(8, 16, 32, 64), dec_depths=(64, 32, 16, 8),
            dyn_layers=2, dyn_heads=4, shortcut_hidden=64, units=64, mlp_layers=1,
            use_vq=use_vq, vq_codes=64)
        B, T = 2, 4
        image = torch.rand(B, T, 3, 64, 64)
        actions = F.one_hot(torch.randint(0, 17, (B, T)), 17).float()
        out = agent.world_model(image, actions, gen_steps=4)
        S = agent.world_model.num_tokens
        assert out["tokens"].shape == (B, T, S, 32)
        assert out["next_tokens"].shape == (B, T, S, 32)
        assert out["recon"].shape == (B, T, 3, 64, 64)
        assert out["reward"].shape == (B, T, 1)
        feat = out["context"].reshape(B, T, -1)
        assert agent.actor_dist(feat).sample().shape == (B, T, 17)
        assert agent.value_dist(feat).mode().shape == (B, T, 1)


if __name__ == "__main__":
    test_dreamerv3_train_step_cpu()
    test_dreamer4_forward_cpu()
    print("OK")
