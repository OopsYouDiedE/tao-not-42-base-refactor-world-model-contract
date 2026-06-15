"""逐 episode 控制重映射 T(in-context「看视频掌握玩法」证明的档①)。

把动作的**语义**打乱、video 不动:模型看到的是真实 Minecraft 画面(真实效果)+ 被重映射
的动作标签。于是「remapped 动作 → 真实效果」这张表 episode 内一致、跨 episode 变化 ⇒ 模型
无法把单一控制方案背进权重,只能从 context(本 window 早期步)做 in-context 系统辨识。
train 与 eval 用 **disjoint** 的置换集 ⇒ holdout 上的表现量的是「从没训练过的控制方案能否靠
观察掌握」,不是记忆。详见 memory incontext-watch-to-master-proof。

动作布局(与 vpt_dataset.VPT_KEYS / train_minecraft.ACT_DIM 严格一致):
    [dx, dy, k0..k19]   N_MOUSE=2,后 20 维键盘。
T 的作用:
  键盘 = 子集置换 π(默认全 20 键;key_subset 给索引则**只置换高信号子集、其余恒等** ⇒
         H=log2(m!) 压进「context 可抽比特」之内,放弃长尾/罕用键的不可辨识野心
         ——12 帧内见证不到的键,信息论上推不出,见 mental_world §6)。train 端拒采落在
         holdout 集里的置换,严格 disjoint(子集机制下 disjoint 只在子集置换上成立)。
         约定 remapped[..., i] = original[..., perm[i]]:重映射方案里第 i 个键 =
         原数据里第 perm[i] 个键的真实效果。
  相机(可选,默认关)= 2 维有符号轴置换:swap(dx↔dy)∈{0,1} × 每轴符号 ∈{±1}。
         键盘是强信号主战场;相机弱信号(见 memory fwd-pred-floor),默认不掺,保持第一枪干净。

签名/单位:apply(act, spec) 中 act 末维 = 22,B 维与 spec 对齐、广播到中间维(act_agg
[B,T-1,22] 与 act_seq [B,T-1,frame_skip,22] 都吃)。键盘置换与「区间 max」、相机有符号
轴置换与「区间平均」都可交换 ⇒ 对 act_seq / act_agg 各自独立施加结果一致。
"""
import math

import torch

N_MOUSE = 2
N_KEYS = 20          # 与 vpt_dataset.VPT_KEYS 一致(11 + 9 hotbar)
ACT_DIM = N_MOUSE + N_KEYS


class ControlRemap:
    """逐 batch 元素(= 逐 episode)采样控制重映射 T,并施加到动作张量上。"""

    def __init__(self, remap_keys=True, remap_camera=False, n_holdout=64, seed=0,
                 key_subset=None):
        self.remap_keys = remap_keys
        self.remap_camera = remap_camera
        # 置换子集:只打乱「高信号、~十几帧内 Δz 可见证」的键(None = 全 20 键,旧行为;
        # 给索引列表则仅置换这些位、其余恒等)。把 H(映射)=log2(m!) 压进「context 可抽比特」
        # 之内 ⇒ 映射在窗口内被观测齐、in-context 辨识良定。holdout/train 两路同构走子集机制
        # ⇒ disjoint 仍只在子集置换上成立。
        self.key_idx = (torch.arange(N_KEYS) if key_subset is None
                        else torch.tensor(sorted({int(i) for i in key_subset}), dtype=torch.long))
        m = int(self.key_idx.numel())
        assert 2 <= m <= N_KEYS, f"key_subset 大小 {m} 须在 [2, {N_KEYS}]"
        assert int(self.key_idx.min()) >= 0 and int(self.key_idx.max()) < N_KEYS, "子集索引越界"
        g = torch.Generator().manual_seed(int(seed))
        # 固定的 holdout 键盘置换 bank(train 端据此拒采 ⇒ 严格 disjoint)。子集小时 m! 有限,
        # holdout 不能吃光置换空间——留至少 1 个给 train(否则拒采死循环)。
        if remap_keys:
            n_ho = max(1, min(int(n_holdout), math.factorial(m) - 1))
            self.holdout_perms = self._sample_subset_perms(n_ho, g)
            self._holdout_set = {tuple(p.tolist()) for p in self.holdout_perms}
        else:
            self.holdout_perms, self._holdout_set = None, set()

    # ------------------------------------------------------------------ 采样
    def _sample_subset_perms(self, B, generator):
        """采 B 个「仅子集互置、其余恒等」的整键置换 [B, N_KEYS]。

        约定与 apply 一致:remapped[..., i] = original[..., perm[i]]。非子集位 perm[i]=i(恒等),
        子集位之间互相置换 ⇒ 整张表只在 m 个高信号键上变化。key_subset=None 时退化为全 20 置换。
        """
        m = int(self.key_idx.numel())
        full = torch.arange(N_KEYS).unsqueeze(0).repeat(B, 1)          # [B,N_KEYS] 恒等
        sub = torch.stack([torch.randperm(m, generator=generator) for _ in range(B)])  # [B,m]
        full[:, self.key_idx] = self.key_idx[sub]                      # 子集位互置,其余不动
        return full

    def _sample_key_perm(self, B, generator, holdout):
        if holdout:
            idx = torch.randint(self.holdout_perms.shape[0], (B,), generator=generator)
            return self.holdout_perms[idx].clone()
        perms = []
        while len(perms) < B:                       # train:拒采落在 holdout 里的置换
            p = self._sample_subset_perms(1, generator)[0]
            if tuple(p.tolist()) not in self._holdout_set:
                perms.append(p)
        return torch.stack(perms)

    def sample(self, B, device="cpu", generator=None, holdout=False):
        """采样 B 个独立 T(每 batch 元素一个 episode)。

        holdout=False ⇒ train 置换(避开 holdout bank);True ⇒ 从固定 holdout bank 抽。
        generator:传一个常数种子的 CPU Generator 可让 eval 每次拿到**同一组**控制方案
        (低方差固定 eval 集,与项目既有固定 eval 哲学一致);train 传 None 用全局 RNG 求多样。
        """
        spec = {}
        if self.remap_keys:
            spec["key_perm"] = self._sample_key_perm(B, generator, holdout).to(device)
        if self.remap_camera:
            spec["cam_swap"] = torch.randint(2, (B,), generator=generator).bool().to(device)
            spec["cam_sign"] = (torch.randint(2, (B, N_MOUSE), generator=generator)
                                * 2 - 1).float().to(device)
        return spec

    # ------------------------------------------------------------------ 施加
    def apply(self, act, spec):
        """act [B, ..., 22] → 重映射后的同形张量(video/特征不受影响)。"""
        assert act.shape[-1] == ACT_DIM, f"末维应为 {ACT_DIM},得到 {act.shape[-1]}"
        B = act.shape[0]
        mid = act.dim() - 2                          # B 与末维之间的中间维数
        mouse, keys = act[..., :N_MOUSE], act[..., N_MOUSE:]
        if "key_perm" in spec:
            perm = spec["key_perm"].view([B] + [1] * mid + [N_KEYS]).expand_as(keys)
            keys = torch.gather(keys, -1, perm)
        if "cam_swap" in spec:
            sw = spec["cam_swap"].view([B] + [1] * mid + [1])
            mouse = torch.where(sw, torch.flip(mouse, dims=[-1]), mouse)
            sign = spec["cam_sign"].view([B] + [1] * mid + [N_MOUSE])
            mouse = mouse * sign
        return torch.cat([mouse, keys], dim=-1)


# ----------------------------------------------------------------------- 自测
if __name__ == "__main__":
    torch.manual_seed(0)
    rm = ControlRemap(remap_keys=True, remap_camera=True, n_holdout=32, seed=7)
    B, T, S = 5, 4, 3

    # 形状 + 键盘置换是「重排」(逐元素集合不变 ⇒ 沿键维求和守恒)
    agg = torch.randn(B, T, ACT_DIM)
    seq = torch.randn(B, T, S, ACT_DIM)
    spec = rm.sample(B)
    agg2, seq2 = rm.apply(agg, spec), rm.apply(seq, spec)
    assert agg2.shape == agg.shape and seq2.shape == seq.shape
    assert torch.allclose(agg2[..., N_MOUSE:].sum(-1), agg[..., N_MOUSE:].sum(-1), atol=1e-5)

    # round-trip:用逆置换 + 逆相机变换还原(证明 T 是双射、无信息丢失)
    perm = spec["key_perm"]
    inv = torch.argsort(perm, dim=-1)
    keys_back = torch.gather(agg2[..., N_MOUSE:], -1, inv.view(B, 1, N_KEYS).expand(B, T, N_KEYS))
    assert torch.allclose(keys_back, agg[..., N_MOUSE:], atol=1e-5), "键盘 round-trip 失败"
    mouse_r = agg2[..., :N_MOUSE] * spec["cam_sign"].view(B, 1, N_MOUSE)   # 符号自逆
    sw = spec["cam_swap"].view(B, 1, 1)
    mouse_back = torch.where(sw, torch.flip(mouse_r, dims=[-1]), mouse_r)
    assert torch.allclose(mouse_back, agg[..., :N_MOUSE], atol=1e-5), "相机 round-trip 失败"

    # train/holdout 严格 disjoint
    tr = rm.sample(200, holdout=False)["key_perm"]
    assert all(tuple(p.tolist()) not in rm._holdout_set for p in tr), "train 采到了 holdout 置换"
    g = torch.Generator().manual_seed(123)
    ho = rm.sample(200, generator=g, holdout=True)["key_perm"]
    assert all(tuple(p.tolist()) in rm._holdout_set for p in ho), "holdout 采样越界"

    # 固定种子 ⇒ eval 每次同一组控制方案(低方差固定 eval 集)
    g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    assert torch.equal(rm.sample(8, generator=g1, holdout=True)["key_perm"],
                       rm.sample(8, generator=g2, holdout=True)["key_perm"])

    # keys-only(默认配置)路径
    rm2 = ControlRemap(remap_keys=True, remap_camera=False)
    s2 = rm2.sample(3)
    assert "cam_swap" not in s2 and torch.allclose(
        rm2.apply(agg[:3], s2)[..., :N_MOUSE], agg[:3, :, :N_MOUSE])

    # 子集置换:仅指定高信号键互置、其余恒等 + 子集 holdout 仍 disjoint
    sub_idx = [0, 1, 2, 3, 4, 7]                        # w/a/s/d/space/attack
    rms = ControlRemap(remap_keys=True, remap_camera=False, n_holdout=16, seed=1,
                       key_subset=sub_idx)
    ss = rms.sample(7)["key_perm"]                      # [7, N_KEYS]
    keep = [i for i in range(N_KEYS) if i not in sub_idx]
    assert torch.equal(ss[:, keep], torch.tensor(keep).unsqueeze(0).expand(7, -1)), \
        "非子集键应恒等"
    assert all(set(p[sub_idx].tolist()) == set(sub_idx) for p in ss), "子集应是子集内的置换"
    assert all(tuple(p.tolist()) not in rms._holdout_set
               for p in rms.sample(50, holdout=False)["key_perm"]), "子集 train 采到 holdout"
    # holdout 不吃光小置换空间(m!−1 上限):2 键只有 2 个置换 ⇒ holdout 截到 1
    assert ControlRemap(key_subset=[0, 1], n_holdout=64).holdout_perms.shape[0] == 1
    print("control_remap 自测全部通过 [OK]")
