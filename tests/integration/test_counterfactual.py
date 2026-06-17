"""集成测试:反事实数据族 + 路径无关 + 反捷径探针(Phase F/G)。

离线 CPU,DI mock 骨干。注意:随机初始化模型**不**具备"按后果分配重要性"的学得行为,
因此这里断言的是**管线接线 + 由构造成立的性质**(数据透传、路径无关损失语义、探针可分性、
eval 指标有限);四类反事实/去相关的**达标线**属训练后的生产验证(见 plan Verification)。
"""
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from domains.minecraft.vpt_dataset import VPTStreamDataset
from net.config import ModelConfig, DynamicsConfig, AdapterConfig, EffectConfig, PredictorConfig
from net.world_model import MinecraftWorldModel
from net.effect_tokenizer import EffectTokenizer
from train.minecraft.train_minecraft import run_sequence
from train.minecraft.eval import evaluate, linear_probe_acc
from train.minecraft.losses import path_invariance_loss
from train.minecraft._seq import _to_float_img, event_segmentation
from blocks.regularization import SIGReg
from tools.download_sample_data import make_counterfactual_set, CF_BRANCHES


class _Mock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed_dim = d
        self.conv = nn.Conv2d(3, d, 8, 8)

    def forward(self, x):
        f = self.conv(x)
        B, d, h, w = f.shape
        return f.view(B, d, h * w).transpose(1, 2)


def _tiny():
    cfg = ModelConfig(d=64, d_rev=48, d_inv=16, act_dim=22, max_skip=2,
                      adapter=AdapterConfig(num_layers=1, nhead=4, ffn_mult=2),
                      dynamics=DynamicsConfig(num_layers=2, nhead=4, ffn_mult=2),
                      effect=EffectConfig(event_vocab_size=8, n_generators=4),
                      predictor=PredictorConfig(n_context_cutoffs=2))
    return MinecraftWorldModel(cfg, backbone=_Mock(64)), cfg


def test_counterfactual_data_roundtrip_and_train():
    """生成反事实族 → 数据集透传 GT → run_sequence(路径项 active)+ eval 指标有限。"""
    tmp = tempfile.mkdtemp(prefix="cf_")
    try:
        make_counterfactual_set(tmp, scenario=0, frames_n=24, size=64, fps=20,
                                seed=0, pixel_energy=0.5)
        make_counterfactual_set(tmp, scenario=1, frames_n=24, size=64, fps=20,
                                seed=1, pixel_energy=0.5)
        ds = VPTStreamDataset(tmp, seq_len=8, fps=20, img_size=32, frame_skip=2,
                              split=None, clip_cache=8, clip_refresh=16)
        batch = next(iter(DataLoader(ds, batch_size=6, num_workers=0)))
        for k in ("has_item", "airborne", "reach_id", "branch"):
            assert k in batch and batch[k].shape[0] == 6, f"缺 GT 字段 {k}"

        model, cfg = _tiny()
        etok = EffectTokenizer(d_inv=cfg.d_inv, event_vocab_size=cfg.effect.event_vocab_size)
        sig = SIGReg(knots=9, num_proj=64)
        bd = {"img": _to_float_img(batch["img"]), "act_agg": batch["act_agg"],
              "dt": batch["dt"], "reach_id": batch["reach_id"][:, 0].long()}
        total, metrics = run_sequence(model.train(), etok, sig, bd, cfg,
                                      beta_sigreg=0.1, beta_guide=0.1, beta_decorr=0.1,
                                      amp_dev="cpu", use_amp=False)
        total.backward()
        assert torch.isfinite(total) and "surprise" in metrics

        ev = evaluate(model, etok, [{"img": batch["img"], "act_agg": batch["act_agg"],
                                     "dt": batch["dt"]}], "cpu", "cpu", False, cfg)
        for k in ("align", "agree", "rollout_drift", "corr_w_future", "corr_w_pixel"):
            assert k in ev and ev[k] == ev[k], f"eval 指标 {k} 非有限"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_path_invariance_semantics():
    """到达同一未来态(reach_id 相同)的样本,ẑ_inv 重合 → 损失为 0;不同则 > 0。"""
    zi = torch.randn(4, 5, 16)
    rid = torch.tensor([0, 0, 1, -1])
    zi_same = zi.clone()
    zi_same[1] = zi[0]                      # 让两个 id=0 的样本 z_inv 重合
    assert float(path_invariance_loss(zi_same, rid)) < 1e-6
    assert float(path_invariance_loss(zi, rid)) > 1e-3


def test_zinv_probe_decodes_separable_signal():
    """探针工具:z_inv 线性可分时高准确率,随机标签≈机会水平。"""
    N, d_inv = 256, 16
    X = torch.randn(N, d_inv)
    y = (X[:, 0] > 0).float()              # 由构造线性可分
    assert linear_probe_acc(X, y, steps=300) > 0.9
    y_rand = (torch.rand(N) > 0.5).float()
    assert linear_probe_acc(X, y_rand, steps=300) < 0.85


def test_event_segmentation_marks_peaks():
    """异常分过阈处置事件边界(Phase E 分段标记)。"""
    score = torch.tensor([[0.0, 0.1, 0.0, 5.0, 0.0, 0.1]])
    b = event_segmentation(score)
    assert b.shape == score.shape and b[0, 3] == 1.0 and b.sum() <= 2


if __name__ == "__main__":
    test_counterfactual_data_roundtrip_and_train()
    test_path_invariance_semantics()
    test_zinv_probe_decodes_separable_signal()
    test_event_segmentation_marks_peaks()
    print("ok")
