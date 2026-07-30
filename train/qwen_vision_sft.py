"""Qwen3.6 视觉 SFT 入口：在 Lumine 动作数据上微调 Qwen3.6 主干。

对外接口：
    main — 命令行入口。

用法::

    python -m train.qwen_vision_sft --model unsloth/Qwen3.6-35B-A3B \
        --dataset-dir runs/datasets/lumine-10xx --output-dir runs/trains/sft-qwen

Qwen3.6 的注意点：原生多模态（Causal LM + Vision Encoder），思考模式是同一份权重的
运行时开关（不再支持 Qwen3 的 ``/think`` 软开关）；35B-A3B 是 MoE（35B 总 / 3B 激活），
不建议 4bit。原生上下文 262144 token，动作预测用不到，按窗口实际长度设
``--max-sequence-length`` 即可。
"""

from __future__ import annotations

from train.command_line import build_argument_parser, run_from_arguments


def main() -> None:
    """解析命令行并跑 Qwen3.6 视觉 SFT。"""
    parser = build_argument_parser(
        description="在 Lumine 动作数据上微调 Qwen3.6 视觉主干",
        default_model="unsloth/Qwen3.6-35B-A3B",
    )
    run_from_arguments(parser.parse_args())


if __name__ == "__main__":
    main()
