#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VPT 原始承包商 jsonl → 事后重标语言目标(hindsight relabel,next_session §2-3)。

动机:BC 期 goal 全零向量,FiLM 通道从未学过"goal 变化 ↔ 行为变化"(GRPO 8×4×2000
run 实证:慢塔指令正确但快塔听不见)。本模块给人类录像倒推出**已发生事实**的语言目标,
让快塔 goal 通道第一次获得真实监督。

事实源(全部来自原始 jsonl 逐帧字段,2026-07-10 对 3 段 6xx 原始数据实测确认;
无人工场景先验、禁 OCR/读屏):
  - ``stats`` 累计计数器逐帧差分:``minecraft.<cat>:minecraft.<item>`` Δ>0 即事件。
    类别→祈使动词是机械映射(见 ``VERB``),宾语 = item id 下划线转空格;
    词表由事件类型机械生成,零手写场景描述。
  - ``isGuiOpen`` 开合沿:开 GUI 事件的容器名取开沿 [0,+1] 帧内的 custom 计数器增量
    (实测对齐窗口;``interact_with_crafting_table``→"crafting table"、``open_chest``→
    "chest" 等,见 ``GUI_STATS``),无增量则是 E 背包 → "inventory";
    关沿 → "close <开沿同名>"。
  - aim(准星/光标像素,0..1000 归一,与慢塔 aim 契约同坐标系):
    非 GUI 帧准星恒在画面中心 (500,500)(渲染事实);GUI 帧光标 = 记录的 mouse.x/y,
    坐标系 1280×720(实测:每次开 GUI 瞬间光标恒 (640,360) = 屏心)。
    标签的 aim 取**事件帧**的准星/光标位置(事件完成时目标就在准星/光标下,是事实;
    "collect"(掉落物飞入)与前溯窗口内目标未对准的帧,aim=事件帧准星只是代理,
    此局限见 knowledge/README.md §3)。

标签规则(唯一窗口启发参数 ``WINDOW``,默认 40 帧 = 2s@20Hz,与 GRPO 慢塔
SLOW_EVERY=20 tick 的指导节奏同量级):帧 f 的标签 = f 之后 WINDOW 帧内最近的事件;
同帧多事件按固定类别优先级 ``PRIORITY`` 取一(craft>mine>kill>gui>pickup>drop>use;
挖掘/合成是意图行为,pickup 多为其结果)。无事件窗口 → 无标签(训练侧 goal 置零)。

goal 向量契约与 grpo_pixel.SlowTower.__call__ 逐字节一致(单测锚定):
    goal[386] = MiniLM(subgoal 文本, normalize_embeddings=True)[384] ⊕ aim/1000 [2]

对外接口:
    frame_facts(raw_lines)  — 原始 jsonl 行 → 逐帧事实 {events, gui, gui_name, cursor}
    label_frames(facts, window) — 事实 → 逐帧 (subgoal, aim) | None
    build_goal(vec384, aim) — 与 SlowTower 同式拼 386 维 goal
    encode_vocab(texts, device) — MiniLM 批量编码(与 grpo_pixel:653 同模型同归一)
    main() — CLI:给既有转换池补拉原始 jsonl 并就地合并标签(幂等,原始转完即删)

CLI(池目录是滚动窗口,下载器可同时在跑;本工具只重写 jsonl、不碰 mp4,
mtime 淘汰按 mp4 排序不受影响):
    python -m train.minecraft.hindsight_relabel --pool runs/data/vpt_early \
        --also runs/data/vpt_holdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"   # 必须与 grpo_pixel 慢塔同模型
GOAL_TEXT_DIM = 384
GOAL_DIM = GOAL_TEXT_DIM + 2
WINDOW = 40                       # 前溯窗口(帧);唯一窗口启发参数,入档声明
CURSOR_WH = (1280.0, 720.0)       # GUI 光标坐标系(实测:开 GUI 瞬间恒 (640,360)=屏心)
CENTER_AIM = (500, 500)           # 非 GUI 准星 = 画面中心(渲染事实)

# stats 类别 → 祈使动词(机械映射;custom 类别除 GUI 命名外全部排除——
# walk_one_cm 等计步器无离散事件语义)
VERB = {"craft_item": "craft", "mine_block": "mine", "kill_entity": "kill",
        "pickup": "collect", "drop": "drop", "use_item": "use"}
# 同帧多事件的固定类别优先级(gui 事件插在 kill 与 pickup 之间)
PRIORITY = ["craft_item", "mine_block", "kill_entity", "gui", "pickup", "drop", "use_item"]
_PRI = {c: i for i, c in enumerate(PRIORITY)}

# GUI 容器命名计数器(minecraft.custom:minecraft.<name>),按列表序取第一个命中
GUI_STATS = ["interact_with_crafting_table", "interact_with_furnace",
             "interact_with_blast_furnace", "interact_with_smoker", "open_chest",
             "open_enderchest", "open_barrel", "open_shulker_box",
             "interact_with_anvil", "interact_with_grindstone",
             "interact_with_smithing_table", "interact_with_stonecutter",
             "interact_with_loom", "interact_with_cartography_table",
             "interact_with_brewingstand", "interact_with_beacon",
             "interact_with_lectern", "inspect_hopper", "inspect_dispenser",
             "inspect_dropper"]


def _gui_noun(stat_name: str) -> str:
    for pfx in ("interact_with_", "open_", "inspect_"):
        if stat_name.startswith(pfx):
            return stat_name[len(pfx):].replace("_", " ")
    return stat_name.replace("_", " ")


def _stat_events(cur: dict, prev: dict) -> list[tuple[str, str]]:
    """两帧 stats → [(类别, item)];只取 VERB 类别的正增量,按优先级+字典序排序。"""
    out = []
    for k, v in cur.items():
        left, _, right = k.partition(":")
        cat = left.rsplit(".", 1)[-1]
        if cat not in VERB:
            continue
        if v - prev.get(k, 0) > 0:
            out.append((cat, right.rsplit(".", 1)[-1].replace("_", " ")))
    out.sort(key=lambda e: (_PRI[e[0]], e[1]))
    return out


def frame_facts(raw_lines: list[str]) -> list[dict]:
    """原始 jsonl 行 → 逐帧事实。与 convert_jsonl 同容错口径:坏行/null = no-op 帧。

    返回每帧 dict:
        events   [(cat, obj)] — stats 差分事件 + GUI 开合事件(("gui","open <名>")等)
        gui      bool
        cursor   (x,y)|None — GUI 帧的原始光标坐标(1280×720 系)
    帧 0 无差分基线,恒无 stats 事件(会话续录时首帧计数器含历史,差分不可信)。
    """
    facts, prev_stats, prev_gui, gui_name = [], None, False, "inventory"
    parsed = []
    for line in raw_lines:
        try:
            d = json.loads(line) if line.strip() else None
        except ValueError:
            d = None
        parsed.append(d if isinstance(d, dict) else {})
    for t, d in enumerate(parsed):
        has_stats = isinstance(d.get("stats"), dict)   # null/坏行缺字段 ⇒ 差分基线顺延
        stats = d["stats"] if has_stats else None
        gui = bool(d.get("isGuiOpen"))
        events = (_stat_events(stats, prev_stats)
                  if has_stats and prev_stats is not None else [])
        if gui and not prev_gui:                     # 开沿:容器名看 [t, t+1] 的 custom 增量
            gui_name = "inventory"
            for dt in (0, 1):
                if t + dt >= len(parsed) or prev_stats is None:
                    break
                st2 = parsed[t + dt].get("stats") or {}
                base = parsed[t + dt - 1].get("stats") or {} if dt else prev_stats
                hit = [n for n in GUI_STATS
                       if st2.get(f"minecraft.custom:minecraft.{n}", 0)
                       - base.get(f"minecraft.custom:minecraft.{n}", 0) > 0]
                if hit:
                    gui_name = _gui_noun(hit[0])
                    break
            events.append(("gui", f"open {gui_name}"))
        elif prev_gui and not gui:                   # 关沿:关的是开沿命名的那个容器
            events.append(("gui", f"close {gui_name}"))
        mouse = d.get("mouse") or {}
        cursor = ((float(mouse.get("x") or 0.0), float(mouse.get("y") or 0.0))
                  if gui else None)
        facts.append({"events": events, "gui": gui, "cursor": cursor})
        prev_stats, prev_gui = (stats if has_stats else prev_stats), gui
    return facts


def _aim_at(fact: dict) -> tuple[int, int]:
    """事件帧的准星/光标 → aim(0..1000,慢塔坐标系)。非 GUI = 画面中心(事实)。"""
    if fact["gui"] and fact["cursor"] is not None:
        x = min(max(fact["cursor"][0] / CURSOR_WH[0], 0.0), 1.0)
        y = min(max(fact["cursor"][1] / CURSOR_WH[1], 0.0), 1.0)
        return int(round(x * 1000)), int(round(y * 1000))
    return CENTER_AIM


def label_frames(facts: list[dict], window: int = WINDOW):
    """逐帧事实 → 逐帧标签 [(subgoal, (ax,ay)) | None]。

    帧 f 的标签 = f 之后 window 帧内**最近**的事件(含 f 自身;同帧多事件按 PRIORITY
    取第一个);subgoal = "<动词> <宾语>",aim = 事件帧的准星/光标。
    """
    n = len(facts)
    labels: list = [None] * n
    for t in range(n):                      # 正序扫描,先到先占 ⇒ 帧 f 拿到最近的下一事件
        ev = facts[t]["events"]
        if not ev:
            continue
        cat, obj = ev[0]
        text = obj if cat == "gui" else f"{VERB[cat]} {obj}"
        aim = _aim_at(facts[t])
        for f in range(max(0, t - window + 1), t + 1):
            if labels[f] is None:
                labels[f] = (text, aim)
    return labels


def build_goal(vec384, aim):
    """与 grpo_pixel.SlowTower 逐字节同式:goal = vec ⊕ aim/1000(torch [386])。"""
    import torch
    v = torch.as_tensor(vec384, dtype=torch.float32)
    return torch.cat([v, torch.tensor([aim[0] / 1000.0, aim[1] / 1000.0])])


def encode_vocab(texts: list[str], device: str = "cpu") -> dict[str, list[float]]:
    """MiniLM 批量编码(与 grpo_pixel:653 同模型、同 normalize_embeddings=True)。"""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_ID, device=device)
    vecs = st.encode(sorted(texts), normalize_embeddings=True)
    return {t: v.tolist() for t, v in zip(sorted(texts), vecs)}


# ────────────────────────────────────────────── 池回填 CLI

def merge_labels(conv_path: str, facts: list[dict], window: int) -> dict:
    """把事实与标签就地合并进转换契约 jsonl(加字段不删字段;原子替换)。

    新增字段(向后兼容,消费方不认识则忽略):
        首行     "relabel_v":1, "window":W
        事件帧   "events": {"cat:obj": n}
        GUI 帧   "cursor": [x,y](1280×720 原始系,留作未来重标不再回源)
        标签帧   "subgoal": str, "aim": [x,y](0..1000)
    返回该段统计。
    """
    with open(conv_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) != len(facts):
        raise RuntimeError(f"帧数不齐 conv={len(lines)} raw={len(facts)}")
    labels = label_frames(facts, window)
    counts: dict[str, int] = {}
    out_lines = []
    for t, (line, fact, lab) in enumerate(zip(lines, facts, labels)):
        rec = json.loads(line)
        if t == 0:
            rec["relabel_v"], rec["window"] = 1, window
        if fact["events"]:
            ev: dict[str, int] = {}
            for cat, obj in fact["events"]:
                k = f"{cat}:{obj}"
                ev[k] = ev.get(k, 0) + 1
            rec["events"] = ev
        if fact["cursor"] is not None:
            rec["cursor"] = list(fact["cursor"])
        if lab is not None:
            rec["subgoal"], rec["aim"] = lab[0], list(lab[1])
            counts[lab[0]] = counts.get(lab[0], 0) + 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    tmp = conv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    os.replace(tmp, conv_path)
    n_lab = sum(counts.values())
    n_gui = sum(1 for x in facts if x["gui"])
    n_lab_nongui = sum(1 for lb, x in zip(labels, facts) if lb and not x["gui"])
    return dict(n=len(lines), n_labeled=n_lab, n_gui=n_gui,
                n_labeled_nongui=n_lab_nongui, vocab=counts)


def _stem_url_map(indexes: list[str]) -> dict[str, str]:
    import requests
    url = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/{}.json"
    m = {}
    for name in indexes:
        try:
            d = requests.get(url.format(name), timeout=60).json()
        except Exception as ex:  # noqa: BLE001 索引失败:跳过,靠其余索引
            print(f"⤫ 索引 {name} 拉取失败: {ex}", flush=True)
            continue
        for rel in d["relpaths"]:
            m[os.path.basename(rel)[:-4]] = d["basedir"] + rel[:-4] + ".jsonl"
    return m


DEFAULT_INDEXES = ("all_6xx_Jun_29,all_7xx_Apr_6,all_8xx_Jun_29,"
                   "all_9xx_Jun_29,all_10xx_Jun_29")


def relabel_dir(pool: str, url_map: dict[str, str], window: int) -> dict:
    """给目录里全部已转换段补拉原始 jsonl 并合并标签(幂等;raw 转完即删)。"""
    import requests
    stems = sorted(f[:-6] for f in os.listdir(pool) if f.endswith(".jsonl"))
    agg = dict(n_clips=0, n_skipped=0, n_frames=0, n_labeled=0, n_gui=0,
               n_labeled_nongui=0, vocab={})
    for i, stem in enumerate(stems):
        conv = os.path.join(pool, stem + ".jsonl")
        try:
            with open(conv, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
        except (OSError, ValueError):
            agg["n_skipped"] += 1
            continue
        if first.get("relabel_v"):
            agg["n_skipped"] += 1              # 已标注:幂等跳过(统计由 scan_vocab 重扫)
            continue
        if stem not in url_map:
            print(f"   ⤫ {stem} 不在索引里,跳过", flush=True)
            agg["n_skipped"] += 1
            continue
        raw = conv + ".raw"
        try:
            with requests.get(url_map[stem], stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(raw, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
            with open(raw, "r", encoding="utf-8") as f:
                facts = frame_facts(f.read().splitlines())
            st = merge_labels(conv, facts, window)
        except Exception as ex:  # noqa: BLE001 网络/半写/被下载器淘汰:跳过不致命
            print(f"   ⤫ {stem}: {ex}", flush=True)
            agg["n_skipped"] += 1
            continue
        finally:
            if os.path.exists(raw):
                os.remove(raw)
        agg["n_clips"] += 1
        for k in ("n_labeled", "n_gui", "n_labeled_nongui"):
            agg[k] += st[k]
        agg["n_frames"] += st["n"]
        for k, v in st["vocab"].items():
            agg["vocab"][k] = agg["vocab"].get(k, 0) + v
        if (i + 1) % 20 == 0 or i + 1 == len(stems):
            print(f"[{i+1}/{len(stems)}] 已标 {agg['n_clips']} 段 "
                  f"覆盖率 {agg['n_labeled']/max(agg['n_frames'],1):.1%}", flush=True)
    return agg


def scan_stats(pool: str) -> dict:
    """全量重扫目录:词表+覆盖率统计(幂等重跑后的唯一事实源,不依赖单次运行增量)。"""
    st = dict(n_clips=0, n_unlabeled_clips=0, n_frames=0, n_labeled=0, n_gui=0,
              n_labeled_nongui=0, vocab={})
    for f in sorted(os.listdir(pool)):
        if not f.endswith(".jsonl"):
            continue
        try:
            with open(os.path.join(pool, f), "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            first = json.loads(lines[0]) if lines else {}
        except (OSError, ValueError, IndexError):
            continue
        if not first.get("relabel_v"):
            st["n_unlabeled_clips"] += 1
            continue
        st["n_clips"] += 1
        for line in lines:
            rec = json.loads(line)
            st["n_frames"] += 1
            gui = bool(rec.get("gui"))
            st["n_gui"] += gui
            sg = rec.get("subgoal")
            if sg:
                st["n_labeled"] += 1
                st["n_labeled_nongui"] += not gui
                st["vocab"][sg] = st["vocab"].get(sg, 0) + 1
    st["coverage"] = round(st["n_labeled"] / max(st["n_frames"], 1), 4)
    st["coverage_nongui"] = round(st["n_labeled_nongui"]
                                  / max(st["n_frames"] - st["n_gui"], 1), 4)
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="给既有 VPT 转换池回填 hindsight 标签")
    ap.add_argument("--pool", default="runs/data/vpt_early")
    ap.add_argument("--also", default="runs/data/vpt_holdout",
                    help="额外目录(holdout);空串跳过")
    ap.add_argument("--indexes", default=DEFAULT_INDEXES)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--vocab-only", action="store_true",
                    help="不回填,只重扫词表并重写 vocab/stats 文件(下载器新段补词表用)")
    args = ap.parse_args()
    pools = [args.pool] + ([args.also] if args.also else [])
    if not args.vocab_only:
        url_map = _stem_url_map([s.strip() for s in args.indexes.split(",") if s.strip()])
        print(f"索引 stem 总数 {len(url_map)}", flush=True)
        for pool in pools:
            print(f"── 回填 {pool}", flush=True)
            agg = relabel_dir(pool, url_map, args.window)
            print(f"   本次新标 {agg['n_clips']} 段 / 跳过 {agg['n_skipped']}", flush=True)
    vocab_all: dict[str, int] = {}
    for pool in pools:                        # 统计与词表都从全量重扫来(幂等重跑安全)
        st = scan_stats(pool)
        st["window"] = args.window
        stats_path = pool.rstrip("/") + "_relabel_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"   ✓ {pool}: {st['n_clips']} 段已标(未标 {st['n_unlabeled_clips']}),"
              f"覆盖率 {st['coverage']:.1%}(非GUI {st['coverage_nongui']:.1%}),"
              f"词表 {len(st['vocab'])} → {stats_path}", flush=True)
        for k, v in st["vocab"].items():
            vocab_all[k] = vocab_all.get(k, 0) + v
    vpath = pools[0].rstrip("/") + "_goal_vocab.json"
    enc = encode_vocab(list(vocab_all), device="cpu")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(enc, f)
    print(f"✅ 词表 {len(enc)} 条(MiniLM 384 维,L2 归一)→ {vpath}", flush=True)


if __name__ == "__main__":
    main()
