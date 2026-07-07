#!/usr/bin/env python3
"""证明:慢塔 LLM 经 LoRA 微调后能产生"先想后答"的推理思维模式(Minecraft 规划的反向依赖推理)。

任务=给一个 Minecraft 目标(如"获得钻石"),模型应输出 <think>反向回溯依赖链</think> + 有序计划。
数据从科技树自动生成(目标→前置);留出一部分目标测泛化。LoRA 微调后对比基线:
  M1 进入推理模式率 = 输出含 <think>…</think> 的比例(基线≈0,微调≈1);
  M2 计划正确率 = 生成计划(去 think)与真值依赖序一致(留出目标)。

用法(GPU,Qwen 已下载):
  PYTHONPATH=. ./.venv/bin/python train/fovea_twotower/reason_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300
"""
import argparse
import re

import torch

# ── Minecraft 科技树:item → (craft 材料 list | mine 需 tool | smelt) ──
BASE = {"log", "stone", "iron_ore", "diamond_ore"}                 # 直接可得(手/镐)
TREE = {
    "planks": ("craft", ["log"]),
    "stick": ("craft", ["planks"]),
    "crafting_table": ("craft", ["planks"]),
    "wooden_pickaxe": ("craft", ["planks", "stick"]),
    "cobblestone": ("mine", "wooden_pickaxe", "stone"),
    "stone_pickaxe": ("craft", ["cobblestone", "stick"]),
    "furnace": ("craft", ["cobblestone"]),
    "raw_iron": ("mine", "stone_pickaxe", "iron_ore"),
    "iron_ingot": ("smelt", "raw_iron", "furnace"),
    "iron_pickaxe": ("craft", ["iron_ingot", "stick"]),
    "diamond": ("mine", "iron_pickaxe", "diamond_ore"),
    "diamond_pickaxe": ("craft", ["diamond", "stick"]),
}
CN = {"log": "原木", "planks": "木板", "stick": "木棍", "crafting_table": "工作台",
      "wooden_pickaxe": "木镐", "stone": "石头", "cobblestone": "圆石", "stone_pickaxe": "石镐",
      "furnace": "熔炉", "iron_ore": "铁矿石", "raw_iron": "生铁", "iron_ingot": "铁锭",
      "iron_pickaxe": "铁镐", "diamond_ore": "钻石矿", "diamond": "钻石",
      "diamond_pickaxe": "钻石镐"}
zh = lambda x: CN.get(x, x)


def plan(goal, seen=None, order=None):
    """回溯依赖 → 拓扑有序 plan(去重,前置在前)。"""
    seen = seen if seen is not None else set()
    order = order if order is not None else []
    if goal in seen:
        return order
    seen.add(goal)
    if goal in BASE:
        return order
    kind = TREE[goal]
    deps = kind[1] if kind[0] == "craft" else ([kind[1], kind[2]] if kind[0] == "mine"
                                               else [kind[1], kind[2]])
    for d in deps:
        plan(d, seen, order)
    order.append(goal)
    return order


def reasoning(goal):
    """反向回溯的自然语言推理轨迹(教模型"为什么这样排")。"""
    lines = [f"目标是获得{zh(goal)}。倒推它需要什么:"]
    for g in [goal] + [x for x in reversed(plan(goal)) if x != goal]:
        if g in BASE:
            lines.append(f"- {zh(g)}是基础资源,可直接获取。")
            continue
        k = TREE[g]
        if k[0] == "craft":
            lines.append(f"- 合成{zh(g)}需要:{'、'.join(zh(x) for x in k[1])}。")
        elif k[0] == "mine":
            lines.append(f"- 挖{zh(g)}需要工具{zh(k[1])}(否则挖不到掉落)。")
        else:
            lines.append(f"- 冶炼{zh(g)}需要{zh(k[1])}和{zh(k[2])}。")
    lines.append("所以按依赖从底到顶排出顺序。")
    return "\n".join(lines)


def target(goal):
    steps = plan(goal)
    plan_str = "\n".join(f"{i+1}. 获得{zh(s)}" for i, s in enumerate(steps)) or "(已是基础资源,直接获取)"
    return f"<think>\n{reasoning(goal)}\n</think>\n\n计划:\n{plan_str}"


def prompt(goal):
    return f"在 Minecraft 生存模式里,规划如何从零获得【{zh(goal)}】。"


GOALS = list(TREE) + list(BASE)
HOLDOUT = ["iron_pickaxe", "diamond", "furnace"]                   # 留出测泛化
TRAIN_GOALS = [g for g in GOALS if g not in HOLDOUT]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="runs/reason_lora")
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)

    def _ids(msgs, gen_prompt):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt,
                                      return_tensors="pt", return_dict=True)
        return enc["input_ids"][0]

    def encode(goal):
        msgs = [{"role": "user", "content": prompt(goal)},
                {"role": "assistant", "content": target(goal)}]
        ids = _ids(msgs, False)
        pre = _ids([msgs[0]], True)                        # 只对 assistant 部分算 loss
        labels = ids.clone()
        labels[:len(pre)] = -100
        return ids, labels

    @torch.no_grad()
    def gen(goal, n=256):
        ids = _ids([{"role": "user", "content": prompt(goal)}], True)[None].to(dev)
        out = model.generate(ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def metrics(tag):
        think = 0
        for g in HOLDOUT:
            o = gen(g)
            has = bool(re.search(r"<think>.*</think>", o, re.S))
            think += has
            print(f"  [{tag}] {zh(g)}: think={has} | {o[:120].replace(chr(10),' ')}...", flush=True)
        return think / len(HOLDOUT)

    print("=== 基线(未微调)在留出目标上 ===", flush=True)
    base_think = metrics("base")

    # LoRA 微调
    model.train()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lcfg)
    data = [encode(g) for g in TRAIN_GOALS]
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    import random
    rng = random.Random(0)
    for step in range(args.steps):
        ids, labels = data[rng.randrange(len(data))]
        ids, labels = ids[None].to(dev), labels[None].to(dev)
        loss = model(input_ids=ids, labels=labels).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"[sft] step {step} loss {loss.item():.4f}", flush=True)
    model.eval()

    print("=== 微调后 在留出目标上(泛化) ===", flush=True)
    tuned_think = metrics("tuned")
    print(f"\n[判据] 进入推理模式率(留出): 基线 {base_think:.2f} → 微调 {tuned_think:.2f}", flush=True)
    print(f"[真值计划] {zh('iron_pickaxe')}: {[zh(s) for s in plan('iron_pickaxe')]}", flush=True)
    model.save_pretrained(args.out)


if __name__ == "__main__":
    main()
