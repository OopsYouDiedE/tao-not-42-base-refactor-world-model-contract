"""以 Gemma 4（MoE VLM）为视觉大模型、直接生成动作 token 的 Minecraft VLA 策略。

对外接口：
    Gemma4PolicyConfiguration — 策略结构与生成配置（纯配置对象）。
    HistoryContext — 一次决策所需的历史帧、任务文本与过去动作。
    build_prompt_messages — 构造 Gemma4 chat 消息（图文交错）。
    Gemma4ActionPolicy — 封装 Gemma4，构造多模态 prompt、生成/解码动作、计算 SFT 损失。
    build_gemma4_action_policy — 用 Unsloth FastVisionModel 加载权重并注入 LoRA。

该模块只负责模型结构与前向，不读取数据集文件、不启动环境（AGENTS §2）。视觉编码由
Gemma4 自带的视觉塔完成，取代旧的 Qwen3VL；动作以文本 token 自回归生成，表示方式由
``net.action_token_codec`` 的 ``ActionTokenFormat`` 决定。Unsloth 仅在
``build_gemma4_action_policy`` 内延迟导入，保持本模块的纯逻辑部分（prompt 构造）可在无
GPU、无 unsloth 的环境下测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from net.action_token_codec import (
    ActionTokenFormat,
    StructuredAction,
    decode_actions,
    describe_format,
    encode_actions,
    encode_single_action,
)


@dataclass(frozen=True)
class Gemma4PolicyConfiguration:
    """Gemma4 动作策略配置。

    Attributes
    ----------
    model_name : str
        Hugging Face / Unsloth 模型标识，默认 unsloth/gemma-4-26B-A4B-it（MoE，Text+Image）。
    action_format : ActionTokenFormat
        动作 token 文本表示。
    action_horizon : int
        一次生成的未来动作帧数；部署只执行前若干步后重规划。
    max_history_frames : int
        prompt 中携带的历史帧数（含当前帧）。
    max_new_tokens : int
        生成动作文本的最大新 token 数。
    """

    model_name: str = "unsloth/gemma-4-26B-A4B-it"
    action_format: ActionTokenFormat = ActionTokenFormat.COMPACT_TAG
    action_horizon: int = 5
    max_history_frames: int = 4
    max_new_tokens: int = 256


@dataclass
class HistoryContext:
    """一次决策的多模态上下文。

    Attributes
    ----------
    frames : list[PIL.Image.Image]
        按时间升序的历史帧（含当前帧），最后一个为当前帧。
    task_text : str
        任务目标英文描述。
    past_actions : list[StructuredAction]
        已执行的历史动作，与除当前帧外的历史帧对应；可为空。
    """

    frames: list[Image.Image]
    task_text: str
    past_actions: list[StructuredAction] = field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are an expert Minecraft player controlling the game frame by frame. "
    "Each frame is shown in time order; the action executed at that frame is given "
    "right after its image. The final frame is the current one and has no action yet. "
    "Predict the sequence of low-level control actions to take next to make progress "
    "on the task. Output only the action lines, nothing else."
)


def build_prompt_messages(
    context: HistoryContext,
    action_format: ActionTokenFormat,
    action_horizon: int,
    include_past_actions: bool = True,
) -> list[dict[str, object]]:
    """构造 Gemma4 chat 消息列表。

    Gemma4Processor 的 chat 模板要求每条消息的 ``content`` 均为块列表（``[{"type": ...}]``），
    裸字符串会触发 ``TypeError``，故 system 也用单块文本列表包裹。

    Parameters
    ----------
    context : HistoryContext
        历史帧、任务与过去动作。
    action_format : ActionTokenFormat
        动作文本格式，决定给模型的说明与样例。
    action_horizon : int
        请模型预测的未来帧数。
    include_past_actions : bool
        是否在 prompt 中给出过去动作文本；实验里用来对比"有正确历史 / 无历史"。

    Returns
    -------
    list[dict]
        可交给 processor.apply_chat_template 的消息列表。
    """
    if not context.frames:
        raise ValueError("context.frames 不能为空")
    if action_horizon < 1:
        raise ValueError("action_horizon 必须大于零")

    frame_count = len(context.frames)
    history_count = frame_count - 1  # 除当前帧外的历史帧数
    # 历史动作与历史帧一一对应；不足则右对齐到离当前帧最近的若干帧。
    past_actions = context.past_actions if include_past_actions else []
    action_by_frame: dict[int, StructuredAction] = {}
    if past_actions:
        offset = history_count - len(past_actions)
        for local_index, action in enumerate(past_actions):
            frame_index = offset + local_index
            if 0 <= frame_index < history_count:
                action_by_frame[frame_index] = action

    # 交错布局：文本锚点夹在图像之间，每帧图像紧跟其时间标签与已执行动作。
    content: list[dict[str, object]] = [
        {"type": "text", "text": f"Task: {context.task_text}."}
    ]
    for frame_index, frame in enumerate(context.frames):
        steps_ago = history_count - frame_index
        if steps_ago > 0:
            label = f"Frame t-{steps_ago}:"
        else:
            label = "Current frame:"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": frame})
        action = action_by_frame.get(frame_index)
        if action is not None:
            content.append({
                "type": "text",
                "text": f"executed: {encode_single_action(action, action_format)}",
            })

    tail = [
        f"Now predict exactly {action_horizon} future action lines, "
        f"labelled t0..t{action_horizon - 1}, in this format:",
        describe_format(action_format),
    ]
    content.append({"type": "text", "text": "\n".join(tail)})
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


class Gemma4ActionPolicy:
    """封装 Gemma4 视觉大模型的动作生成与监督接口。

    只做前向 / 生成 / 监督损失，不涉及数据加载与环境交互。生成路径解码为定长结构化动作
    块，监督路径只对目标动作 token 计交叉熵（图像与 prompt token 标签置 -100）。
    """

    def __init__(self, model, processor, configuration: Gemma4PolicyConfiguration):
        self.model = model
        self.processor = processor
        self.configuration = configuration

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _apply_chat(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        """把消息经 processor 转成模型输入张量并搬到设备。"""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in inputs.items()}

    def _apply_chat_batch(
        self,
        conversations: list[list[dict[str, object]]],
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        """批量把多条对话转成左 padding 的模型输入张量并搬到设备。"""
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in inputs.items()}

    def generate_actions(
        self,
        context: HistoryContext,
        include_past_actions: bool = True,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> tuple[list[StructuredAction], str]:
        """自回归生成并解码为定长动作块，返回(动作块, 模型原始文本)。"""
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(self.model)
        messages = build_prompt_messages(
            context, self.configuration.action_format,
            self.configuration.action_horizon, include_past_actions,
        )
        inputs = self._apply_chat(messages, add_generation_prompt=True)
        if seed is not None:
            torch.manual_seed(seed)
        generation_arguments: dict[str, object] = {
            "max_new_tokens": self.configuration.max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_arguments["temperature"] = float(temperature)
        generated = self.model.generate(**inputs, **generation_arguments)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_length:]
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True,
        )[0]
        actions = decode_actions(text, self.configuration.action_horizon)
        return actions, text

    def supervised_loss(
        self,
        contexts: list[HistoryContext],
        targets: list[list[StructuredAction]],
    ) -> torch.Tensor:
        """对一个 micro-batch 计算 SFT 交叉熵：只监督目标动作 token。

        每个样本把目标动作序列作为 assistant 轮加入对话，批量走完整对话模板编码；再批量
        编码"仅到生成起点"的 prompt 取各样本 prompt 长度。processor 用左 padding，故动作
        token 一律右对齐——每样本 ``动作 token 数 = 完整有效长度 - prompt 有效长度``，据此把
        末尾这些位置设为监督标签、其余（system/user/图像/padding）置 -100（AGENTS §5 只对
        动作监督）。batch 内图像尺寸一致，pixel_values 可张量化。
        """
        if len(contexts) != len(targets):
            raise ValueError("contexts 与 targets 数量必须一致")
        if not contexts:
            raise ValueError("batch 不能为空")
        from unsloth import FastVisionModel

        FastVisionModel.for_training(self.model)
        full_conversations = []
        prompt_conversations = []
        for context, target_actions in zip(contexts, targets):
            messages = build_prompt_messages(
                context, self.configuration.action_format,
                len(target_actions), include_past_actions=True,
            )
            target_text = encode_actions(target_actions, self.configuration.action_format)
            full_conversations.append(messages + [
                {"role": "assistant",
                 "content": [{"type": "text", "text": target_text}]},
            ])
            prompt_conversations.append(messages)
        full_inputs = self._apply_chat_batch(full_conversations, add_generation_prompt=False)
        prompt_inputs = self._apply_chat_batch(prompt_conversations, add_generation_prompt=True)
        full_valid = full_inputs["attention_mask"].sum(dim=1)  # 每样本非 pad token 数
        prompt_valid = prompt_inputs["attention_mask"].sum(dim=1)
        action_lengths = (full_valid - prompt_valid).tolist()
        sequence_length = full_inputs["input_ids"].shape[1]
        labels = torch.full_like(full_inputs["input_ids"], -100)
        for row, action_length in enumerate(action_lengths):
            if action_length <= 0:
                raise RuntimeError("目标动作 token 数非正，检查模板与目标文本")
            start = sequence_length - int(action_length)
            labels[row, start:] = full_inputs["input_ids"][row, start:]
        outputs = self.model(**full_inputs, labels=labels)
        return outputs.loss


def build_gemma4_action_policy(
    configuration: Gemma4PolicyConfiguration,
    device: torch.device,
    cache_directory: str | None = None,
    lora_rank: int = 0,
) -> Gemma4ActionPolicy:
    """用 Unsloth FastVisionModel 加载 Gemma4 权重与 processor 并构造动作策略。

    Parameters
    ----------
    configuration : Gemma4PolicyConfiguration
        策略配置（模型名、格式、horizon 等）。
    device : torch.device
        目标设备（须为支持 BF16 的 CUDA）。
    cache_directory : str | None
        Hugging Face 缓存目录；None 用默认。
    lora_rank : int
        >0 时注入 LoRA 适配器（仅训语言层的注意力与 MLP，含 MoE 专家）。=0 用于纯推理。

    Returns
    -------
    Gemma4ActionPolicy
    """
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=configuration.model_name,
        load_in_4bit=False,
        load_in_16bit=True,
        use_gradient_checkpointing="unsloth",
        use_exact_model_name=True,
        cache_dir=cache_directory,
    )
    if lora_rank > 0:
        model = FastVisionModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.0,
            bias="none",
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            random_state=0,
        )
    return Gemma4ActionPolicy(model, processor, configuration)
