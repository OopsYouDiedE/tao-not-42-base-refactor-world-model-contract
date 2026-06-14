"""Minecraft 世界模型的解码头:动作规划(未来)与逆动力学(过去)。

对外接口:
    N_CAMERA_BINS       — 相机 mu-law 分箱数(必须与 domains.minecraft.vpt_action.CAMERA_BINS
                          一致;net 层不 import domain 层,数值在训练端校验)。
    DecoderHeads        — DETR 式未来 K 步定时动作计划(onset/键/鼠标分箱)。
    InverseDynamicsHead — 从潜变化 ΔZ 反推已发生动作(槽路 + 可选 patch 旁路 / ctx-FiLM)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CAMERA_BINS = 11   # 与 domains.minecraft.vpt_action.CAMERA_BINS 一致(net 层不 import domain,训练端校验)


class DecoderHeads(nn.Module):
    """Minecraft 复合动作解码器：同时预测鼠标移动和大量键盘按键。"""
    def __init__(self, d, num_keyboard_keys=20, n_cam_bins=N_CAMERA_BINS):
        super().__init__()
        self.n_cam_bins = n_cam_bins
        # 鼠标预测 (dx, dy 回归)
        self.mouse_head = nn.Sequential(
            nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 2)
        )
        # 键盘预测 (20 个独立二分类)
        self.keyboard_head = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, num_keyboard_keys)
        )

        # 动作规划头(DETR 式,**未来动作**输出)。时间契约:onset 的 0 时刻 =
        # 本次 forward 传入的 t_vec 时刻("现在"),单位 = **帧**(与 dt/t_hist 同)。
        # 监督 = 行为克隆(train 的 plan_bc_loss):槽 k 对齐未来第 k+1 个
        # 转移的聚合动作与时长。鼠标用 mu-law 分箱 logits 而非回归——回归 MSE 的
        # "恒 0"与无权重 CE 的基率不动点都在 inv-dyn 上踩过坑,这里直接用终案。
        self.plan_mouse = nn.Linear(d, 2 * n_cam_bins)
        self.plan_keyboard = nn.Linear(d, num_keyboard_keys)
        self.plan_onset = nn.Linear(d, 1)             # 多久后按下(帧,自"现在"起算)
        self.plan_dur = nn.Linear(d, 1)               # 按住多久(帧)
        self.plan_exist = nn.Linear(d, 1)             # 该计划槽是否有效

    def decode_action_plan(self, u_tokens):
        """预测未来长程动作计划。onset 累积 softplus 参数化 ⇒ 沿查询维单调,
        消除查询置换对称(与 tao 版 DecoderHeads.decode_action_plan 同理),
        也使"槽 k ↔ 时间序上第 k 个未来转移"的对齐无需匈牙利匹配。"""
        B, K = u_tokens.shape[:2]
        onset_inc = F.softplus(self.plan_onset(u_tokens)).squeeze(-1)   # [B,K] 正增量
        return {
            "mouse_logits": self.plan_mouse(u_tokens).view(B, K, 2, self.n_cam_bins),
            "keyboard": torch.sigmoid(self.plan_keyboard(u_tokens)),
            "onset": torch.cumsum(onset_inc, dim=1),
            "duration": F.softplus(self.plan_dur(u_tokens)).squeeze(-1),
            "exist": torch.sigmoid(self.plan_exist(u_tokens)).squeeze(-1)
        }


class InverseDynamicsHead(nn.Module):
    """从潜变化 ΔZ 反推动作。鼠标 = mu-law 分箱**分类**(logits),键盘 = 20 独立二分类。

    鼠标弃回归改分类的原因:dx/dy 分布是"尖峰在 0 + 重尾大转身",MSE 下恒预测 0
    即近似最优(上一版实测 mouse loss 钉死在边缘方差处一动不动);分类目标下
    基率解的 CE = 边缘熵,任何真实信号都能压过它。分箱定义在 domains.minecraft.vpt_action
    (camera_to_bin/bin_to_camera),与 VPT 原版的 mu-law 离散相机一致。
    """
    def __init__(self, d, num_keyboard_keys=20, n_cam_bins=N_CAMERA_BINS, enc_dim=None,
                 use_ctx=False):
        super().__init__()
        self.n_cam_bins = n_cam_bins
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()
        )
        self.mouse_out = nn.Linear(64, 2 * n_cam_bins)
        self.kb_out = nn.Linear(64, num_keyboard_keys)
        # 上下文条件化(in-context「看视频掌握玩法」):用脑内记忆 h(已 attend 过去动作与
        # 潜变化 ⇒ 携带本 episode 的 (动作→效果) 历史)FiLM 调制 Δz→动作的读出特征。
        # **乘性**调制(γ⊙feat+β,非仅加偏置)才能表达「按本局 in-context 推断的控制映射 T
        # 去解读同一个 Δz」;纯偏置只能挪 logits、表达不了「同样的效果在不同控制下对应不同键」。
        # 零初始化 ⇒ 冷启动恒等(ctx 无效、退化为原 context-blind 头),有利可图才被打开;
        # inv-dyn 损失的梯度经 FiLM 回流进 h 的生成 ⇒ 逼 h 把「这一局的控制映射」编码进去。
        self.use_ctx = use_ctx
        if use_ctx:
            self.ctx_film = nn.Linear(d, 2 * 64)
            nn.init.zeros_(self.ctx_film.weight)
            nn.init.zeros_(self.ctx_film.bias)
        # 全 patch 平均 Δz 支路(oracle 复盘新增):本地 oracle 阶梯实测,从冻结 DINOv2
        # 的**全 patch 平均** Δz 直接读键盘可达 onset 0.34 / bal 0.81,而经 16 槽池化+c
        # 门控后只剩 0.15——信息在槽池化处丢了。这条支路绕开槽瓶颈,直接吃 patch 平均
        # Δz(就是 oracle 的 pool 口径),与槽路 logits **相加**:槽路仍拿 inv-dyn 梯度
        # (c 门控的唯一梯度来源不变),patch 路补回被池化丢掉的读出精度。enc_dim=None
        # 时退化为原行为(无 patch 路),向后兼容。
        self.use_patch = enc_dim is not None
        if self.use_patch:
            self.patch_net = nn.Sequential(
                nn.Linear(enc_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU()
            )
            self.patch_mouse = nn.Linear(64, 2 * n_cam_bins)
            self.patch_kb = nn.Linear(64, num_keyboard_keys)

    def forward(self, residual_z, patch_dz=None, ctx=None):
        # residual_z: [B, N, d](槽-Δz·c);patch_dz: [B, enc_dim](全 patch 平均 Δz);
        # ctx: [B, d] 脑内记忆 h(use_ctx 时由调用方传 pre-step h ⇒ 不含当前动作、无泄漏)。
        # 返回 (槽路 mouse_logits[B,2,bins], 槽路 kb_prob[B,keys], parts)。
        # ⚠ 两路 logits **不再相加**(2026-06-14):加法融合下损失只看「和」,patch 旁路
        # (直读冻结 patch-mean Δz、信号更干净)会先把目标解释掉,残差→0 把槽路+c 的梯度
        # 掐断(gradient starvation)。这里只返回槽路预测(= 喂世界模型/c、rollout 唯一可用
        # 的诚实读出);patch 旁路 logits 经 parts 单独抛出,由损失侧独立监督(参数互斥 ⇒
        # 梯度不回流槽路),纯作"patch-mean Δz 可读出多少"的天花板诊断。
        feat = self.net(residual_z.mean(dim=1))
        if self.use_ctx and ctx is not None:
            g, b = self.ctx_film(ctx.to(feat.dtype)).chunk(2, dim=-1)
            feat = feat * (1.0 + g) + b               # 零初始化 ⇒ 起步恒等
        mouse_logits = self.mouse_out(feat).view(-1, 2, self.n_cam_bins)
        kb_logit = self.kb_out(feat)
        parts = None
        if self.use_patch and patch_dz is not None:
            pf = self.patch_net(patch_dz)             # patch 旁路不吃 ctx:纯「无-context」诊断基线
            parts = (self.patch_mouse(pf).view(-1, 2, self.n_cam_bins), self.patch_kb(pf))
        return mouse_logits, torch.sigmoid(kb_logit), parts
