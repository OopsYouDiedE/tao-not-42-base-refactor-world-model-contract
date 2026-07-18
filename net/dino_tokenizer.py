"""DINO-tokenizer:冻结 DINOv3/v2 骨干 + 可训练空间卷积解码头(net/dino_tokenizer.py)。

对外接口:
    DinoTokenizer — 冻结视觉骨干出 patch 特征网格,轻量 ConvDecoder 解回像素,只训解码头。

设计意图(见 knowledge/README.md §2.1):感知先验借**冻结 DINOv3**
(不从零训编码器),但世界模型需**可解码隐空间**做重建/流匹配,故在冻结 patch 特征上另训一个
小解码头。相较 net/dreamer4.Tokenizer 的从零 ConvEncoder,这里:
  · 编码器零训练、no_grad 前向(不存反向激活,便宜),故可放心用高分辨率取更细 patch 网格;
  · 解码头**直接吃 [B,enc_dim,G,G] 空间网格**逐级上采样,**不经巨型 fc**
    (net/dreamer4 那个 Linear(num_tokens·token_dim, ...) 随 img⁴ 暴涨、易 OOM,这里根除)。

patch 网格:G = img_size / patch_size(DINOv3 patch=16;img=176→G=11)。骨干 token 布局
= [CLS, register×n_reg, patch×G²],patch 特征取 last_hidden_state[:, 1+n_reg:]。
"""
import torch
from torch import nn

from blocks.conv import Conv2dSamePad, ImgChLayerNorm
from net.backbone import load_backbone

# ImageNet 归一化(DINO 预训练所用;与 net/bc/policy.py 一致)
_PX_MEAN = (0.485, 0.456, 0.406)
_PX_STD = (0.229, 0.224, 0.225)


class SpatialConvDecoder(nn.Module):
    """空间网格 [B,in_dim,G,G] → 图像 [B,out_ch, G·stride^L, ·]。无 fc,逐级 Upsample+Conv。

    与 blocks/decoder.py::ConvDecoder 的区别:输入已是空间特征图(骨干 patch 网格),
    故省去把展平隐向量投回小图的 Linear;上采样级数 L=len(depths),输出边长 = G·stride^L。
    """

    def __init__(self, in_dim, depths, out_channels=3, kernel=5, stride=2,
                 act=nn.SiLU, norm=True, upsample_mode="nearest"):
        super().__init__()
        self.depths = tuple(depths)
        self.proj = Conv2dSamePad(in_dim, self.depths[0], 1, stride=1)  # 1×1 投影
        layers, c = [], self.depths[0]
        for d in self.depths[1:]:
            layers.append(nn.Upsample(scale_factor=stride, mode=upsample_mode))
            layers.append(Conv2dSamePad(c, d, kernel, stride=1, bias=not norm))
            if norm:
                layers.append(ImgChLayerNorm(d))
            layers.append(act())
            c = d
        # 末级:再上采样一次 → 映射到图像通道,不接归一化/激活(输出原始像素)
        layers.append(nn.Upsample(scale_factor=stride, mode=upsample_mode))
        layers.append(Conv2dSamePad(c, out_channels, kernel, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, grid):
        return self.layers(self.proj(grid))


class DinoTokenizer(nn.Module):
    """冻结 DINO 骨干 + 空间卷积解码头。forward(image)→(recon, feats)。

    Parameters
    ----------
    kind : str
        骨干种类,"dinov3"(gated)或 "dinov2"(开放)。
    dec_depths : tuple[int]
        解码头逐级通道;级数须使 G·2^len == img_size(如 G=11、img=176→len=4)。
    weights : str | None
        覆盖骨干 HF repo id(可换更大 ViT)。

    Attributes
    ----------
    patch_size, enc_dim, n_register : int
        骨干属性;grid_side = img_size // patch_size 由 forward 时输入决定。
    """

    def __init__(self, kind="dinov3", dec_depths=(384, 256, 128, 64), weights=None):
        super().__init__()
        module, patch, enc_dim, n_reg = load_backbone(kind, weights)
        self.backbone = module.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)                    # 冻结:不训编码器
        self.patch_size = patch
        self.enc_dim = enc_dim
        self.n_register = n_reg
        self.decoder = SpatialConvDecoder(enc_dim, dec_depths)
        self.register_buffer("px_mean", torch.tensor(_PX_MEAN).view(1, 3, 1, 1))
        self.register_buffer("px_std", torch.tensor(_PX_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def encode(self, image):
        """冻结骨干出 patch 网格。image [B,3,H,W] ∈[0,1] → feats [B,enc_dim,G,G]。"""
        h, w = image.shape[-2:]
        g_h, g_w = h // self.patch_size, w // self.patch_size
        xn = (image - self.px_mean) / self.px_std      # ImageNet 归一化
        tokens = self.backbone(pixel_values=xn).last_hidden_state  # [B,1+n_reg+G²,C]
        patches = tokens[:, 1 + self.n_register:]      # 去 CLS + register → [B,G²,C]
        return patches.reshape(-1, g_h, g_w, self.enc_dim).permute(0, 3, 1, 2)

    def forward(self, image):
        """image [B,3,H,W] ∈[0,1] → (recon [B,3,H,W] ∈[0,1], feats [B,enc_dim,G,G])。"""
        feats = self.encode(image)
        recon = self.decoder(feats)
        return recon, feats
