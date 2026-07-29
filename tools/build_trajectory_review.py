"""为 MineStudio 轨迹候选题生成可离线翻阅的 HTML 审核册。"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat


TASK_LABELS = {
    "demonstration_optimization": "演示优化",
    "image_sequence_to_action": "图像序列反推动作",
    "history_to_future_action": "历史图像预测动作",
    "single_frame_intent_to_action": "单帧意图转动作",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def image_metrics(paths: list[Path]) -> dict[str, float]:
    brightness: list[float] = []
    contrast: list[float] = []
    edge_energy: list[float] = []
    arrays: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            grayscale = rgb.convert("L")
            brightness.append(float(mean(ImageStat.Stat(rgb).mean)))
            contrast.append(float(ImageStat.Stat(grayscale).stddev[0]))
            edge_energy.append(float(ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES)).mean[0]))
            arrays.append(np.asarray(rgb, dtype=np.float32))
    changes = [
        float(np.abs(right - left).mean())
        for left, right in zip(arrays, arrays[1:])
    ]
    return {
        "minimum_brightness": min(brightness),
        "mean_brightness": mean(brightness),
        "minimum_contrast": min(contrast),
        "mean_edge_energy": mean(edge_energy),
        "maximum_visual_change": max(changes, default=0.0),
    }


def screening_flags(task_type: str, metrics: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if metrics["minimum_brightness"] < 16.0:
        flags.append("偏暗，需人工确认可辨识性")
    if metrics["minimum_contrast"] < 8.0:
        flags.append("对比度偏低，需人工确认画面信息")
    if task_type == "image_sequence_to_action" and metrics["maximum_visual_change"] < 1.5:
        flags.append("连续画面变化较弱，需人工确认动作可反推性")
    return flags


def source_name(sample_id: str, boundary: int) -> str:
    sequence = int(sample_id.rsplit("_", 1)[1])
    return "10xx" if sequence < boundary else "7xx"


def escaped_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def write_contact_sheets(
    dataset: Path,
    questions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    directory: Path,
    questions_per_sheet: int = 12,
) -> list[str]:
    """把统计预警题按行排成视觉复核联系表。"""
    attention = {
        record["id"]: record for record in records
        if record["screening_status"] == "needs_attention"
    }
    selected = [question for question in questions if question["id"] in attention]
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    thumbnail_size = (160, 90)
    row_height = 128
    for sheet_index, offset in enumerate(range(0, len(selected), questions_per_sheet), 1):
        batch = selected[offset:offset + questions_per_sheet]
        canvas = Image.new("RGB", (1000, 28 + row_height * len(batch)), "#f4f1ea")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"MineStudio attention review {sheet_index}", fill="#202020")
        for row_index, question in enumerate(batch):
            top = 28 + row_index * row_height
            record = attention[question["id"]]
            draw.rectangle((5, top + 2, 995, top + row_height - 3), outline="#aaa399")
            draw.text((12, top + 8), question["id"], fill="#202020")
            draw.text((12, top + 25), "; ".join(record["flags"]), fill="#8a321f")
            for image_index, relative in enumerate(question["images"]):
                with Image.open(dataset / relative) as source:
                    thumbnail = source.convert("RGB")
                    thumbnail.thumbnail(thumbnail_size)
                left = 175 + image_index * 163
                image_top = top + 35
                canvas.paste(thumbnail, (left, image_top))
                draw.text((left, top + 18), f"frame {question['source']['image_frames'][image_index]}", fill="#4d4943")
        path = directory / f"attention_{sheet_index:02d}.jpg"
        canvas.save(path, quality=92)
        outputs.append(str(path))
    return outputs


def build_review(dataset: Path, output: Path, source_boundary: int = 100) -> dict[str, Any]:
    questions = read_jsonl(dataset / "questions.jsonl")
    answers = {item["id"]: item for item in read_jsonl(dataset / "answer_key.jsonl")}
    records: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, question in enumerate(questions, 1):
        sample_id = question["id"]
        task_type = question["task_type"]
        answer = answers[sample_id]
        paths = [dataset / relative for relative in question["images"]]
        metrics = image_metrics(paths)
        flags = screening_flags(task_type, metrics)
        source = source_name(sample_id, source_boundary)
        record = {
            "id": sample_id,
            "source_dataset": source,
            "task_type": task_type,
            "automatic_structure_check": "pass",
            "screening_status": "needs_attention" if flags else "pending_visual_review",
            "flags": flags,
            "image_metrics": {key: round(value, 3) for key, value in metrics.items()},
            "formal_review_status": "pending_human_and_ai_review",
        }
        records.append(record)
        image_tags = "".join(
            f'<figure><a href="{html.escape(relative)}" target="_blank">'
            f'<img loading="lazy" src="{html.escape(relative)}" alt="{html.escape(sample_id)}">'
            f'</a><figcaption>帧 {frame}</figcaption></figure>'
            for relative, frame in zip(question["images"], question["source"]["image_frames"])
        )
        inputs = question.get("inputs", {})
        input_blocks = ""
        if inputs.get("intent"):
            input_blocks += f'<h4>给定意图</h4><p>{html.escape(inputs["intent"])}</p>'
        if inputs.get("raw_action_sequence"):
            input_blocks += f'<h4>待优化原始动作</h4><pre>{escaped_json(inputs["raw_action_sequence"])}</pre>'
        flag_html = "".join(f"<li>{html.escape(flag)}</li>" for flag in flags) or "<li>未触发统计预警</li>"
        cards.append(f"""
<article class="card" data-id="{html.escape(sample_id)}" data-source="{source}"
 data-task="{html.escape(task_type)}" data-screen="{record['screening_status']}">
  <header><span class="number">#{index}</span><h2>{html.escape(sample_id)}</h2>
    <span class="badge source">{source}</span><span class="badge">{TASK_LABELS[task_type]}</span>
    <span class="badge screen">{record['screening_status']}</span></header>
  <div class="images">{image_tags}</div>
  <div class="columns"><section><h3>题目</h3><p>{html.escape(question['prompt'])}</p>
    {input_blocks}<h4>来源</h4><pre>{escaped_json(question['source'])}</pre></section>
  <section><h3>参考答案</h3><pre>{escaped_json(answer['reference_action_sequence'])}</pre>
    <h4>统计预警</h4><ul>{flag_html}</ul><pre>{escaped_json(record['image_metrics'])}</pre></section></div>
  <div class="review"><label>决定
    <select data-field="decision"><option value="pending">待审核</option><option value="approve">通过</option>
      <option value="reject">拒绝</option><option value="revise">修订</option></select></label>
    <label>理由或修改意见<textarea data-field="notes" rows="3"></textarea></label></div>
</article>""")

    output.parent.mkdir(parents=True, exist_ok=True)
    screening_path = output.with_suffix(".jsonl")
    screening_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    sheets = write_contact_sheets(dataset, questions, records, output.with_name("review_sheets"))
    counts = Counter(record["screening_status"] for record in records)
    task_options = "".join(
        f'<option value="{task}">{label}</option>' for task, label in TASK_LABELS.items()
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MineStudio 轨迹题审核册</title><style>
:root{{--bg:#f4f1ea;--paper:#fffdf8;--ink:#26231f;--muted:#6e665d;--line:#d8d0c4;--accent:#315c4a;--warn:#9b4d2f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
.toolbar{{position:sticky;top:0;z-index:5;padding:14px 18px;background:#263c34;color:white;box-shadow:0 2px 10px #0004}}
.toolbar h1{{display:inline;margin:0 24px 0 0;font-size:20px}}select,input,button,textarea{{font:inherit}}
.toolbar select,.toolbar input,.toolbar button{{margin:4px;padding:7px;border:0;border-radius:5px}}
.summary{{max-width:1500px;margin:18px auto;padding:12px 18px;background:var(--paper);border:1px solid var(--line)}}
#cards{{max-width:1500px;margin:auto}}.card{{margin:18px;padding:18px;background:var(--paper);border:1px solid var(--line);border-radius:8px}}
.card header{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}}.card h2{{margin:0;font-size:18px}}.number{{color:var(--muted)}}
.badge{{padding:3px 8px;border-radius:99px;background:#e5ebe7;color:#29483b;font-size:12px}}.badge.source{{background:#e8e2d7}}
.images{{display:flex;gap:10px;overflow-x:auto;margin:14px 0;padding-bottom:8px}}figure{{margin:0;min-width:220px}}
figure img{{display:block;width:100%;height:150px;object-fit:contain;background:#171717;border-radius:4px}}figcaption{{text-align:center;color:var(--muted)}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}pre{{white-space:pre-wrap;word-break:break-word;background:#f0ede6;padding:10px;border-radius:5px;font-size:12px}}
.review{{display:grid;grid-template-columns:180px 1fr;gap:12px;border-top:1px solid var(--line);padding-top:14px}}label{{display:flex;flex-direction:column;gap:5px}}
.review textarea,.review select{{padding:7px;border:1px solid var(--line);border-radius:4px;background:white}}.hidden{{display:none}}ul{{padding-left:20px}}
@media(max-width:800px){{.columns,.review{{grid-template-columns:1fr}}figure{{min-width:180px}}}}
</style></head><body>
<div class="toolbar"><h1>MineStudio 轨迹题审核册</h1>
<select id="source"><option value="">全部来源</option><option>10xx</option><option>7xx</option></select>
<select id="task"><option value="">全部题型</option>{task_options}</select>
<select id="screen"><option value="">全部初筛状态</option><option value="needs_attention">需重点查看</option><option value="pending_visual_review">待视觉审核</option></select>
<select id="decision"><option value="">全部审核决定</option><option value="pending">待审核</option><option value="approve">通过</option><option value="reject">拒绝</option><option value="revise">修订</option></select>
<input id="query" placeholder="搜索 ID"><button id="export">导出审核 JSON</button></div>
<div class="summary">共 {len(records)} 道；统计预警 {counts['needs_attention']} 道；未触发预警 {counts['pending_visual_review']} 道。
本页面的决定保存在当前浏览器。点击图片可查看原图；导出文件可交给后续审核流程，但不自动视为正式双审结果。当前显示 <b id="visible">{len(records)}</b> 道。</div>
<main id="cards">{''.join(cards)}</main>
<script>
const cards=[...document.querySelectorAll('.card')];
const source=document.getElementById('source'), task=document.getElementById('task');
const screen=document.getElementById('screen'), decision=document.getElementById('decision');
const query=document.getElementById('query'), visible=document.getElementById('visible');
const key='minestudio-trajectory-review-v1'; const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
for(const card of cards){{const state=saved[card.dataset.id]||{{decision:'pending',notes:''}};card.querySelector('[data-field=decision]').value=state.decision;card.querySelector('[data-field=notes]').value=state.notes;}}
function persist(){{const state={{}};for(const card of cards)state[card.dataset.id]={{decision:card.querySelector('[data-field=decision]').value,notes:card.querySelector('[data-field=notes]').value}};localStorage.setItem(key,JSON.stringify(state));filter();}}
function filter(){{let count=0;for(const card of cards){{const ok=(!source.value||card.dataset.source===source.value)&&(!task.value||card.dataset.task===task.value)&&(!screen.value||card.dataset.screen===screen.value)&&(!decision.value||card.querySelector('[data-field=decision]').value===decision.value)&&(!query.value||card.dataset.id.includes(query.value.trim()));card.classList.toggle('hidden',!ok);if(ok)count++;}}visible.textContent=count;}}
document.querySelectorAll('.review select,.review textarea').forEach(node=>node.addEventListener('change',persist));
[source,task,screen,decision,query].forEach(node=>node.addEventListener('input',filter));
document.getElementById('export').onclick=()=>{{const result=cards.map(card=>({{id:card.dataset.id,decision:card.querySelector('[data-field=decision]').value,notes:card.querySelector('[data-field=notes]').value,review_kind:'human_browser_workbook'}}));const blob=new Blob([result.map(x=>JSON.stringify(x)).join('\n')+'\n'],{{type:'application/jsonl'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='browser_reviews.jsonl';link.click();URL.revokeObjectURL(link.href);}};
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    return {
        "sample_count": len(records),
        "needs_attention": counts["needs_attention"],
        "pending_visual_review": counts["pending_visual_review"],
        "html": str(output),
        "screening": str(screening_path),
        "contact_sheets": sheets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MineStudio 轨迹题离线审核册")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-boundary", type=int, default=100)
    arguments = parser.parse_args()
    print(json.dumps(
        build_review(arguments.dataset_dir, arguments.output, arguments.source_boundary),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
