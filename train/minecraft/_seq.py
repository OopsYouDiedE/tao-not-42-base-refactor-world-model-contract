"""训练/评估共用的低层序列与张量助手(无状态,纯 torch)。

放这里是为了打断循环依赖:train_minecraft(引擎)与 eval 都用这些 helper,
而 main 又调用 eval ⇒ helper 不能住在任何一边。
"""
import torch


def _to_float_img(img):
    """uint8 [.,3,H,W] → float∈[0,1](归一化推迟到 GPU 上做,PCIe 传 uint8 省 4×)。"""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()


def event_segmentation(score, threshold=None):
    """按异常/surprise 分数切分序列(Phase E:先落地分段标记,token 合并默认关)。

    score: [B, T] 逐帧异常分(surprise × z_inv 持续性等)。分数过阈处置 1 = 事件边界;
    可预测波谷(低分连续段)→ 可合并成延展 token(时间抽象),波峰 → 事件 token。
    threshold=None 时取每条序列均值 + 标准差作自适应阈。返回 boundary 掩码 [B, T](float 0/1)。
    """
    s = score.float()
    if threshold is None:
        threshold = s.mean(dim=-1, keepdim=True) + s.std(dim=-1, keepdim=True)
    return (s > threshold).float()
