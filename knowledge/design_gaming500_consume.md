# 设计：如何消费 gaming500-720p-hdf5（原生动作形态，不套 VPT 契约）

> 2026-07-03。回答"下游训练怎么吃 `unjustify/gaming500-720p-hdf5`"。
> 关联：生产管线见 [design_gaming500_hd_pretrain.md](design_gaming500_hd_pretrain.md) §9
> （HDF5 归档），本文只谈**读取端**。落地代码：`train/gaming500/dataset.py`。
> 结论先行：**动作按原生形态露出（原始像素 dx/dy、20 位键掩码、frame_idx 变率对齐），
> 不硬套 `train/minecraft` 的 VPT 22 维契约**；归一化尺度留给各训练目标按自身分布标定。

## 0. 定位：这是 500h 数据的"编码后内容"

`unjustify/gaming500-720p-hdf5` 是 `markov-ai/gaming-500-hours` 原始录像经
`tests/encode_gaming500_hdf5.py` 编码后的**随机访问帧库**：图像 JPEG 入 HDF5、动作事件
全率保留。**动作的原始物理范围（dx/dy 单位、键名、gui）以源数据集为准**——
源 `frame_events.json` 的 schema 未在卡片上明示，由 `tests/convert_gaming500.py::frame_actions`
反解（鼠标绝对坐标帧内差分和、回中跳变按 `warp_px` 丢弃；实测 `|dx| p99≈75 px`）。
要精确标定归一化尺度时读源数据集，不在本层臆测。

## 1. 仓库现状（2026-07-03 实测）

| 项 | 值 |
|---|---|
| 分片 | `shard_0000.h5` … `shard_0014.h5`，共 15 片 |
| 总量 | 148.2 GB（单片 7.5–11 GB），**公开** |
| 附属 | 仅 `.gitattributes`——**无 `manifest.json`**（manifest 只在生产机本地） |

⇒ 读取端不能依赖 manifest，必须自己扫描分片建段索引（`Gaming500Dataset.__init__` 即如此）。

## 2. HDF5 布局回顾（每个游戏段一组，权威定义见 encode 脚本 §9）

```
/{game}/{session8}_{seg:02d}/
    jpeg       vlen u8 [N]   # JPEG 字节流,15Hz;attrs: hz / w=1280 / h=720 / quality=80
    frame_idx  i32  [N]      # 每张图在本段 30Hz 事件流里的下标(offset ∈ [0, seg_end-seg_start))
    dx, dy     f32  [M]      # 相机位移,单位=原始像素,30Hz 全率;M = seg_end-seg_start
    keys       u32  [M]      # 20 键位掩码,bit i = 编码期键序第 i 个(见 §4 KEY_NAMES)
    gui        u8   [M]      # 是否在菜单
    events_gz  u8   [K]      # 整段逐帧契约 JSONL 的 gzip 原文(无损兜底)
    attrs: task / session / game / seg_start / seg_end / src_fps=30 / meta_json
```

关键不变量：**图像 15Hz、动作 30Hz，`frame_idx` 是二者唯一对齐桥**。
`frame_idx` 与 `dx/dy/keys` 同处 `[0, M)` 偏移空间（encode 端 `frame_idx=off`、
`dx/dy/keys=acts[s:e]`），所以第 j 张图的即时动作 = `dx[frame_idx[j]]`。

## 3. 为什么不和 VPT 强行对齐（用户 2026-07-03 拍板）

`train/minecraft` 的 VPT 契约（`vpt_action.py` / `vpt_dataset.py`）是 **Minecraft 专用**，
把它套到多游戏 gaming500 上会静默注入三重错误先验：

1. **键语义先验**：VPT 20 键含 `key_inventory` / `key_sneak` / `key_sprint` /
   `key_hotbar.1..9`——这些是 Minecraft 键位。gaming500 覆盖 valorant / gta-v /
   cod / hitman 等，同一 bit 在不同游戏语义不一致；当成"库存键"训练是错的。
2. **聚合先验**：`VPTStreamDataset._split_actions` 鼠标取**区间均值**（= 平均角速度）。
   gaming500 图像 15Hz、事件 30Hz，一步跨 2 个 30Hz 帧；对**位移量**取均值会把
   "两帧转了多少"抹成"每帧转多快"，多游戏不同灵敏度下尤其失真。位移的守恒量是**和**。
3. **尺度先验**：`CAMERA_SCALE`/`CAMERA_MU` 按 BASALT 相机"度"校准（转身 ±190）。
   gaming500 的 dx/dy 是**原始像素**、逐游戏鼠标灵敏度不同，单一尺度必错配。

**原生策略（本文契约）**：
- dx/dy **保原始像素**，区间聚合取**和**（位移守恒）；不做 `CAMERA_SCALE` 归一化——
  归一化是**训练目标的事**，各目标按自己看到的分布（源范围见 §0）现场标定。
- keys 只当**20 位无语义掩码**，区间内"按过即 1"（OR）；键名词表(§4)仅作数据事实登记，
  不承诺任何游戏的语义映射。
- dt 由 `frame_idx` 差分给出**真实时距**（单位：30Hz 帧），下游要秒就 `/30`。
- 图像默认方形 resize，另供 center/random crop（兑现 design §3 的原生密度裁剪流）。

## 4. 原生样本契约（`Gaming500Dataset` 输出）

序列窗口样本（`seq_len=L`）：

| 键 | Shape | Dtype | 单位/含义 |
|---|---|---|---|
| `img` | `[L, 3, H, W]` | uint8 | RGB，H=W=`img_size`（720p 缩放/裁剪） |
| `dx`,`dy` | `[L-1]` | f32 | 相邻图像间**位移和**，原始像素（未归一化） |
| `keys` | `[L-1, 20]` | uint8 | 区间内 OR 的键掩码 multihot |
| `gui` | `[L-1]` | uint8 | 区间内是否进过菜单 |
| `dt` | `[L-1]` | int32 | 区间跨的 30Hz 帧数（真实时距 = dt/30 s） |
| `game`,`task` | `str` | | 段 attrs |

`seq_len=1`（**tokenizer 单帧模式**）：只出 `img [1,3,H,W]` + `game`/`task`，
不做动作对齐——这是 design §6 阶段 1（多尺度重建）的直供路径。

`KEY_NAMES`（20，bit 位序 = 编码期写入序，复刻为**数据事实**，不引入 Minecraft 语义）：
`key_w,key_a,key_s,key_d,key_space,key_sneak,key_sprint,key_attack,key_use,key_drop,`
`key_inventory,key_hotbar.1..9`。

## 5. 变率对齐（15Hz 图 / 30Hz 事件）

第 j→j+1 张图之间，30Hz 子帧区间为 `(frame_idx[j], frame_idx[j+1]]`：

| 量 | 聚合 | 理由 |
|---|---|---|
| dx,dy | `sum` over 子帧 | 位移可加，和 = 该区间总转动/移动 |
| keys | `bitwise OR` | "区间内按过"是控制信号的自然语义 |
| gui | `any` | 菜单态在区间内出现即标记 |
| dt | `frame_idx[j+1]-frame_idx[j]` | 图像采样非严格等距，dt 显式携带真实跨度 |

下游若要变间隔训练，自采样 `Δt` 后仍用上表聚合——聚合语义与采样间隔解耦。

## 6. 三种消费模式（对应 design 各阶段）

- **tokenizer 预训练**（阶段 1）：`seq_len=1`，`crop_mode="random"` 混 `"resize"`，
  只吃图像；720p 原生密度喂细节（design §7 闸 1）。
- **dynamics 预训练**（阶段 3）：`seq_len=16+`，原生动作序列做条件；训练目标内部
  自行归一化 dx/dy、自选是否用 20 位键。**不经 VPT 层**。
- **原生高频裁剪流**：`crop_mode="center"/"random"` 从 720p 直裁，不缩放平均，
  保任务相关高频（准星/字体/纹理）——design §7 的"不经缩放平均的通道"。

## 7. 落地与依赖方向

- 代码落 `train/gaming500/dataset.py`（AGENTS §8：数据集区分压在 `train/<域>` 层，
  与 `train/minecraft` 并列自洽，**不 import 它**——多游戏域不继承 Minecraft 契约）。
- 底层分片读取（h5py 打开 / JPEG 解码 / 段索引）在本模块内自足，**不依赖
  `tests/encode_gaming500_hdf5.py::Gaming500H5`**（tests 是离线/编码端，不得作生产依赖）。
  二者是"编码端冒烟读取"与"训练端加载器"两个职责；本加载器是训练侧 SSOT。
- 分片获取：`huggingface_hub.hf_hub_download` 按需拉单片到 `runs/data/g500_h5/`
  （runs/ 已 gitignore）；`Gaming500Dataset(root)` 扫描目录里所有 `.h5`。

## 8. 风险与开口

- **源范围未落地**：dx/dy 的官方单位/上界仍需读 `markov-ai/gaming-500-hours` 原文
  确认（§0）；当前只有 convert 脚本的经验分位数（`|dx| p99≈75`）。
- **键词表跨游戏语义**：20 位掩码在非 Minecraft 游戏语义不明；tokenizer 阶段无关，
  dynamics 阶段若发现某些位在多游戏下是噪声，考虑按游戏 mask 或降权。
- **损坏/未封分片**：仓库无 manifest，个别分片可能截断；加载器逐片 try-open、
  跳损坏片并告警（同 encode 端策略）。
- **裁剪位置策略**：random vs 运动能量加权裁剪待小实验（承 design §8）。
