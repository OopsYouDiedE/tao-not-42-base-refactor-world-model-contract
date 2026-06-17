"""离线诊断评估：词表预测准确率与特征重建误差 (两步式离散词表与重构版)。
"""
import torch
import torch.nn.functional as F

from train.minecraft.losses import vocab_pred_loss, z_recon_loss
from train.minecraft._seq import roll_hist, _to_float_img


@torch.no_grad()
def evaluate(model, action_tok, loader, device, amp_dev, use_amp):
    """在 holdout 数据集上测试词表准确率与特征重建误差。"""
    model.eval()
    action_tok.eval()

    total_loss = 0.0
    total_vocab_acc = 0.0
    total_recon_ratio = 0.0
    n_steps = 0

    for batch in loader:
        img = _to_float_img(batch["img"].to(device))
        act_seq = batch["act_seq"].to(device)
        dt = batch["dt"].to(device)
        t_vec = batch["t_vec"].to(device)
        task_emb = batch.get("task_emb")
        task_emb = task_emb.to(device) if task_emb is not None else None
        
        B, T = img.shape[0], img.shape[1]
        
        h = torch.zeros(B, 1, model.d, device=device)
        a_hist = torch.zeros(B, model.J, act_seq.shape[-1], device=device)
        t_hist = torch.zeros(B, model.J, device=device)
        hv = torch.zeros(B, model.J, device=device)

        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
            z_tg = model.encode_obs(feats=feats).view(B, T, -1, model.d)
        feats = feats.view(B, T, *feats.shape[-2:])
        z_tg = z_tg.float()

        for t in range(T - 1):
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                # 1. 提取词表 GT
                valid = (torch.arange(model.S, device=device).unsqueeze(0) < dt[:, t].unsqueeze(1)).float()
                _, target_token_id, tok_loss = action_tok(act_seq[:, t], valid_mask=valid)
                
                # 2. 模型前向
                out = model(
                    z_tg[:, t], h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                    t_hist=t_hist, hist_valid=hv, task_emb=task_emb,
                    target_token_id=target_token_id
                )

            agg_action = act_seq[:, t].mean(dim=1)
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, agg_action, dt[:, t])

            logits = out["logits"].float()
            z_recon = out["z_recon"].float()

            l_vocab = vocab_pred_loss(logits, target_token_id)
            l_recon, r_recon = z_recon_loss(z_recon, z_tg[:, t + 1])
            l_tok = tok_loss.mean()

            total_loss += (l_vocab + l_recon + l_tok).item()
            total_vocab_acc += (logits.argmax(dim=-1) == target_token_id).float().mean().item()
            total_recon_ratio += r_recon
            n_steps += 1
            h = out["h_next"]

    return {
        "loss": total_loss / max(n_steps, 1),
        "vocab_acc": total_vocab_acc / max(n_steps, 1),
        "recon_ratio": total_recon_ratio / max(n_steps, 1)
    }
