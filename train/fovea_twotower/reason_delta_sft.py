#!/usr/bin/env python3
"""E2:慢脑"库存差额规划"知识注入——规则引擎无限生成监督对,验内容正确率(非只验思维模式)。

命题(用户设想的慢脑职责:判断局势/规划收集/核查差多少):MC 知识是可执行规则,
最快注入路径 = 程序生成 (库存状态, 正确差额计划) 对做 SFT,而非喂百科文本。
reason_sft 已证思维模式可诱导(留出 think 率 0→1.00)但内容正确性未过关(13 条样本);
本脚本把样本换成组合生成的"给定库存,补齐到目标"任务,并用**多解拓扑序校验器**
判内容正确(接受任何合法顺序,不逐字对答案)。

预登记判据(先于结果):
  M2a 见过目标 × 新库存组合:正确率 ≥ 0.8;
  M2b 留出目标(训练从未见,任意库存):正确率 ≥ 0.6。
  基线(未微调)双指标对照必报。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/reason_delta_sft.py \
      --model Qwen/Qwen2.5-1.5B-Instruct --n_train 400 --steps 800
"""
import argparse
import hashlib
import random
import re

import torch

from train.fovea_twotower.reason_sft import BASE as BASE0
from train.fovea_twotower.reason_sft import CN as CN0
from train.fovea_twotower.reason_sft import TREE as TREE0

# ── 扩展科技树(E2 首轮教训:9 个训练目标 → loss 300 步塌 0 = 背诵;M2b 0.37。
#    "反向依赖展开"这个程序要泛化,需要目标多样性,不是步数)──
BASE = BASE0 | {"coal_ore", "copper_ore", "gold_ore", "redstone_ore", "sand"}
TREE = dict(TREE0) | {
    "coal": ("mine", "wooden_pickaxe", "coal_ore"),
    "torch": ("craft", ["stick", "coal"]),
    "raw_copper": ("mine", "stone_pickaxe", "copper_ore"),
    "copper_ingot": ("smelt", "raw_copper", "furnace"),
    "raw_gold": ("mine", "iron_pickaxe", "gold_ore"),
    "gold_ingot": ("smelt", "raw_gold", "furnace"),
    "golden_pickaxe": ("craft", ["gold_ingot", "stick"]),
    "redstone": ("mine", "iron_pickaxe", "redstone_ore"),
    "wooden_axe": ("craft", ["planks", "stick"]),
    "wooden_sword": ("craft", ["planks", "stick"]),
    "stone_axe": ("craft", ["cobblestone", "stick"]),
    "stone_sword": ("craft", ["cobblestone", "stick"]),
    "iron_axe": ("craft", ["iron_ingot", "stick"]),
    "iron_sword": ("craft", ["iron_ingot", "stick"]),
    "diamond_sword": ("craft", ["diamond", "stick"]),
    "diamond_axe": ("craft", ["diamond", "stick"]),
    "chest": ("craft", ["planks"]),
    "ladder": ("craft", ["stick"]),
    "bowl": ("craft", ["planks"]),
    "bucket": ("craft", ["iron_ingot"]),
    "shears": ("craft", ["iron_ingot"]),
    "iron_door": ("craft", ["iron_ingot"]),
    "minecart": ("craft", ["iron_ingot"]),
    "rail": ("craft", ["iron_ingot", "stick"]),
    "hopper": ("craft", ["iron_ingot", "chest"]),
    "shield": ("craft", ["planks", "iron_ingot"]),
    "anvil": ("craft", ["iron_ingot"]),
    "glass": ("smelt", "sand", "furnace"),
    "glass_bottle": ("craft", ["glass"]),
    "blast_furnace": ("craft", ["furnace", "iron_ingot"]),
    "campfire": ("craft", ["stick", "coal", "log"]),
    "lantern": ("craft", ["torch", "iron_ingot"]),
    "compass": ("craft", ["iron_ingot", "redstone"]),
    "piston": ("craft", ["planks", "cobblestone", "iron_ingot", "redstone"]),
    "dropper": ("craft", ["cobblestone", "redstone"]),
    "lever": ("craft", ["stick", "cobblestone"]),
    "redstone_torch": ("craft", ["stick", "redstone"]),
}
CN = dict(CN0) | {
    "coal_ore": "煤矿石", "coal": "煤炭", "torch": "火把", "copper_ore": "铜矿石",
    "raw_copper": "粗铜", "copper_ingot": "铜锭", "gold_ore": "金矿石",
    "raw_gold": "粗金", "gold_ingot": "金锭", "golden_pickaxe": "金镐",
    "redstone_ore": "红石矿石", "redstone": "红石", "wooden_axe": "木斧",
    "wooden_sword": "木剑", "stone_axe": "石斧", "stone_sword": "石剑",
    "iron_axe": "铁斧", "iron_sword": "铁剑", "diamond_sword": "钻石剑",
    "diamond_axe": "钻石斧", "chest": "箱子", "ladder": "梯子", "bowl": "碗",
    "bucket": "铁桶", "shears": "剪刀", "iron_door": "铁门", "minecart": "矿车",
    "rail": "铁轨", "hopper": "漏斗", "shield": "盾牌", "anvil": "铁砧",
    "sand": "沙子", "glass": "玻璃", "glass_bottle": "玻璃瓶",
    "blast_furnace": "高炉", "campfire": "营火", "lantern": "灯笼",
    "compass": "指南针", "piston": "活塞", "dropper": "投掷器", "lever": "拉杆",
    "redstone_torch": "红石火把",
}
zh = lambda x: CN.get(x, x)  # noqa: E731  遮蔽 reason_sft.zh(它闭包旧 CN)

HOLDOUT_GOALS = ["iron_pickaxe", "diamond", "furnace",
                 "hopper", "lantern", "piston", "golden_pickaxe"]
ALL_GOALS = [g for g in TREE if g not in HOLDOUT_GOALS]
EN = {v: k for k, v in CN.items()}


def deps_of(item):
    k = TREE[item]
    return list(k[1]) if k[0] == "craft" else [k[1], k[2]]


def missing_plan(goal, inv):
    """给定库存 → 缺失项的依赖有序补齐计划(拥有某物=其子树全免)。"""
    order, seen = [], set()

    def rec(g):
        if g in seen or g in inv or g in BASE:
            return
        seen.add(g)
        for d in deps_of(g):
            rec(d)
        order.append(g)
    rec(goal)
    return order


def check_plan(goal, inv, steps):
    """多解校验:集合=真缺失集,且每步前置被(库存∪BASE∪已完成)覆盖。"""
    true_set = set(missing_plan(goal, inv))
    if set(steps) != true_set:
        return False
    have = set(inv) | set(BASE)
    for s in steps:
        if s not in TREE or not all(d in have for d in deps_of(s)):
            return False
        have.add(s)
    return goal in have or goal in inv or goal in BASE


def sample_inv(goal, rng):
    """随机库存:真依赖链的随机子集 + 随机无关项 + 15% 含目标本身。

    C1b 教训:曾因 `x != goal` 过滤,"任务已完成"态从不在分布内 → 慢脑拿着生铁
    仍说缺生铁(复核环节全灭)。"核查差多少"必须会判"不差了"。"""
    chain = missing_plan(goal, set())
    inv = {x for x in chain if rng.random() < 0.4 and x != goal}
    inv |= {x for x in rng.sample(list(TREE), 3) if rng.random() < 0.3 and x != goal}
    if rng.random() < 0.15:
        inv.add(goal)
    return frozenset(inv)


def recipe_card():
    """全配方参考表(≈600 token)。运行时慢脑带着配方书工作——留出目标的规则
    只能来自上下文,不可能凭空回忆(E2b 教训:M2b 无卡=考回忆未见规则,0.37 是必然)。"""
    lines = ["参考配方表:"]
    for it in sorted(TREE):
        k = TREE[it]
        if k[0] == "craft":
            lines.append(f"- {zh(it)}=合成:{'、'.join(zh(x) for x in k[1])}")
        elif k[0] == "mine":
            lines.append(f"- {zh(it)}=开采:{zh(k[2])}(需{zh(k[1])})")
        else:
            lines.append(f"- {zh(it)}=冶炼:{zh(k[1])}(需{zh(k[2])})")
    return "\n".join(lines)


def prompt(goal, inv, card=False):
    inv_s = "、".join(zh(x) for x in sorted(inv)) if inv else "(空)"
    head = (recipe_card() + "\n\n") if card else ""
    return (f"{head}Minecraft 生存模式。当前库存:{inv_s}。目标:获得【{zh(goal)}】。"
            f"核查还缺什么,给出按依赖顺序的补齐计划;不缺则答\"已齐备\"。")


def reasoning(goal, inv, steps):
    lines = [f"目标{zh(goal)}。核查库存:已有 {('、'.join(zh(x) for x in sorted(inv))) or '无'}。"]
    for s in reversed(steps):
        k = TREE[s]
        need = "、".join(zh(d) for d in deps_of(s))
        verb = {"craft": "合成", "mine": "开采", "smelt": "冶炼"}[k[0]]
        lines.append(f"- 缺{zh(s)}:{verb}它需要 {need}。")
    if inv:
        lines.append("库存里已有的部分不必重做。")
    return "\n".join(lines)


def target(goal, inv):
    steps = missing_plan(goal, inv)
    if not steps:
        return "<think>\n核查库存:目标或其全部前置已具备。\n</think>\n\n已齐备。"
    plan_s = "\n".join(f"{i+1}. 获得{zh(s)}" for i, s in enumerate(steps))
    return f"<think>\n{reasoning(goal, inv, steps)}\n</think>\n\n计划:\n{plan_s}"


def parse_steps(text):
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = []
    for m in re.finditer(r"\d+\.\s*(?:获得)?([\w一-鿿]+)", body):
        it = EN.get(m.group(1).strip(), m.group(1).strip())
        out.append(it)
    return out


def is_split_test(goal, inv):
    """库存组合按哈希切 15% 做"新库存"测试组(与训练组不相交,确定性)。"""
    h = hashlib.md5(f"{goal}|{'|'.join(sorted(inv))}".encode()).hexdigest()
    return int(h, 16) % 100 < 15


def gen_dataset(n, rng):
    data, seen = [], set()
    goals = ALL_GOALS
    while len(data) < n:
        g = rng.choice(goals)
        inv = sample_inv(g, rng)
        if (g, inv) in seen or is_split_test(g, inv):
            continue
        seen.add((g, inv))
        data.append((g, inv))
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_eval", type=int, default=30)
    p.add_argument("--out", default="runs/reason_delta_lora")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)

    def _ids(msgs, gen_prompt):
        enc = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=gen_prompt,
                                      return_tensors="pt", return_dict=True)
        return enc["input_ids"][0]

    def encode(g, inv, card=False):
        msgs = [{"role": "user", "content": prompt(g, inv, card)},
                {"role": "assistant", "content": target(g, inv)}]
        ids = _ids(msgs, False)
        pre = _ids([msgs[0]], True)
        labels = ids.clone()
        labels[:len(pre)] = -100
        return ids, labels

    @torch.no_grad()
    def gen(g, inv, card=False, n=384):
        ids = _ids([{"role": "user", "content": prompt(g, inv, card)}],
                   True)[None].to(dev)
        out = model.generate(ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    # 评测组:M2a 见过目标×新库存(哈希测试组);M2b 留出目标
    r2 = random.Random(999)
    eva = []
    while len(eva) < args.n_eval:
        g = r2.choice(ALL_GOALS)
        inv = sample_inv(g, r2)
        if is_split_test(g, inv):
            eva.append((g, inv))
    evb = []
    while len(evb) < args.n_eval:
        g = r2.choice(HOLDOUT_GOALS)
        evb.append((g, sample_inv(g, r2)))

    def metrics(tag):
        res = {}
        for name, group, card in (("M2a_新库存", eva, False),
                                  ("M2a_带卡", eva, True),
                                  ("M2b_留出目标", evb, False),
                                  ("M2b_带卡", evb, True)):
            ok = 0
            for g, inv in group:
                o = gen(g, inv, card)
                steps = parse_steps(o)
                good = (check_plan(g, set(inv), steps) if missing_plan(g, inv)
                        else ("已齐备" in o and not steps))
                ok += good
            res[name] = ok / len(group)
            print(f"  [{tag}] {name}: {res[name]:.2f}", flush=True)
        return res

    print("=== 基线(未微调) ===", flush=True)
    base = metrics("base")

    model.train()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    data = [encode(g, inv, card=rng.random() < 0.5)
            for g, inv in gen_dataset(args.n_train, rng)]
    print(f"[sft] {len(data)} 样本(库存差额任务,组合生成,半数带配方卡)", flush=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr)
    for step in range(args.steps):
        ids, labels = data[rng.randrange(len(data))]
        loss = model(input_ids=ids[None].to(dev), labels=labels[None].to(dev)).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"[sft] step {step} loss {loss.item():.4f}", flush=True)
    model.eval()

    print("=== 微调后 ===", flush=True)
    tuned = metrics("tuned")
    print(f"\n[判据] M2a≥0.8: {tuned['M2a_新库存']:.2f} "
          f"({'PASS' if tuned['M2a_新库存'] >= 0.8 else 'FAIL'}) | "
          f"M2b带卡≥0.6: {tuned['M2b_带卡']:.2f} "
          f"({'PASS' if tuned['M2b_带卡'] >= 0.6 else 'FAIL'}) | "
          f"基线 {base}", flush=True)
    model.save_pretrained(args.out)


if __name__ == "__main__":
    main()
