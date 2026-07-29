"""从生成数据集中导出不含参考答案的模型盲测请求与回答模板。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.minestudio_finetune.review_questions import read_jsonl


def prepare_model_eval(dataset_directory: Path, output_directory: Path) -> dict[str, Any]:
    """导出请求；图片继续引用数据集 images，避免复制二进制文件。"""
    output_directory.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(dataset_directory / "questions.jsonl")
    requests_path = output_directory / "requests.jsonl"
    template_path = output_directory / "responses_template.jsonl"
    with requests_path.open("w", encoding="utf-8") as requests_handle, template_path.open(
        "w", encoding="utf-8",
    ) as template_handle:
        for question in questions:
            request = {
                "id": question["id"],
                "task_type": question["task_type"],
                "prompt": question["prompt"],
                "images": [f"../{path}" for path in question["images"]],
                "inputs": question["inputs"],
                "output_contract": question["output_contract"],
            }
            requests_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            template_handle.write(json.dumps({
                "id": question["id"],
                "answer": [],
            }, ensure_ascii=False) + "\n")
    readme = """# 模型盲测包

`requests.jsonl` 每行是一道独立视觉题。模型必须读取 `images` 中按顺序排列的全部图片，
按照 `prompt` 作答，并把结果写入 `responses.jsonl`。回答格式与
`responses_template.jsonl` 相同：`id` 原样返回，`answer` 是一个或多个变长动作块组成的数组。

本目录不包含参考答案。不要向做题模型提供上级目录中的 `answer_key.jsonl`、生成报告或审核文件。

完成答题后运行：

```bash
python -m datasets.minestudio_finetune.test_answers \
  --dataset-dir datasets/minestudio_finetune/luna_eval \
  --responses datasets/minestudio_finetune/luna_eval/blind/responses.jsonl \
  --output datasets/minestudio_finetune/luna_eval/luna_results.jsonl
```

报告三个口径：动作协议格式通过率、与人类参考轨迹的相似度分布、视觉语义审核通过率。
由于动作存在多解，最终正确率采用视觉语义审核通过率；参考相似度只作为诊断指标。
"""
    (output_directory / "README.md").write_text(readme, encoding="utf-8")
    return {
        "questions": len(questions),
        "requests": str(requests_path),
        "response_template": str(template_path),
        "contains_answer_key": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="导出不含答案的视觉模型盲测包")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(
        prepare_model_eval(arguments.dataset_dir, arguments.output_dir),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
