"""训练/评估共用的低层序列与张量助手(无状态,纯 torch)。

放这里是为了打断循环依赖:train_minecraft(引擎)与 eval 都用 roll_hist / _to_float_img,
而 main 又调用 eval ⇒ 两个 helper 不能住在任何一边。
"""
import torch


def roll_hist(a_hist, t_hist, hv, action, dt_cur):
    """时间前进 dt_cur 帧并滚入一个刚结束的聚合动作。

    a_hist [B,J,A] / t_hist [B,J](各条目结束时刻距"现在"的帧数)/ hv [B,J] 有效位。
    旧条目统一变老 dt_cur 帧;新条目刚结束 ⇒ age=0、有效位=1。开头的空槽
    (a_hist 初始全零)有效位为 0——全零动作是合法的"没按",不能用值判空。
    """
    B = action.shape[0]
    z1 = torch.zeros(B, 1, device=action.device, dtype=t_hist.dtype)
    return (torch.cat([a_hist[:, 1:], action.unsqueeze(1)], dim=1),
            torch.cat([t_hist[:, 1:] + dt_cur.unsqueeze(1), z1], dim=1),
            torch.cat([hv[:, 1:], torch.ones_like(z1)], dim=1))


def _to_float_img(img):
    """uint8 [.,3,H,W] → float∈[0,1](归一化推迟到 GPU 上做,PCIe 传 uint8 省 4×)。"""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()
