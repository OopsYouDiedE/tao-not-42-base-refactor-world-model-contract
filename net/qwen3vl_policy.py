"""以 Qwen3VL 为视觉大模型、直接生成动作 token 的 Minecraft VLA 策略。

对外接口：
    Qwen3VLPolicyConfiguration — 策略结构与生成配置（纯配置对象）。
    HistoryContext — 一次决策所需的历史帧、任务文本与过去动作。
    Qwen3VLActionPolicy — 封装 Qwen3VL，构造多模态 prompt、生成/解码动作、计算 SFT 损失。
    build_qwen3vl_action_policy — 加载权重并构造策略（可选 LoRA）。

该模块只负责模型结构与前向，不读取数据集文件、不启动环境（AGENTS §2）。视觉编码由
Qwen3VL 自带的视觉塔完成，取代旧的 DINOv3 快塔；动作以文本 token 自回归生成，表示
方式由 ``net.action_token_codec`` 的 ``ActionTokenFormat`` 决定。
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
class Qwen3VLPolicyConfiguration:
    """Qwen3VL 动作策略配置。

    Attributes
    ----------
    model_name : str
        Hugging Face 模型标识，默认 Qwen3-VL-8B-Instruct。
    action_format : ActionTokenFormat
        动作 token 文本表示。
    action_horizon : int
        一次生成的未来动作帧数；部署只执行前若干步后重规划。
    max_history_frames : int
        prompt 中携带的历史帧数（含当前帧）。
    image_max_pixels : int
        单帧送入视觉塔前的像素上限，控制显存与序列长度。
    max_new_tokens : int
        生成动作文本的最大新 token 数。
    """

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    action_format: ActionTokenFormat = ActionTokenFormat.COMPACT_TAG
    action_horizon: int = 5
    max_history_frames: int = 4
    image_max_pixels: int = 256 * 28 * 28
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
    """构造 Qwen3VL chat 消息列表。

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
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class Qwen3VLActionPolicy:
    """封装 Qwen3VL 视觉大模型的动作生成与监督接口。

    该类持有 processor 与 causal LM 主体。生成路径把多模态 prompt 编码后自回归解码为
    动作文本，再由 codec 转成结构合法的动作块；监督路径用 teacher forcing 只对目标
    动作 token 计交叉熵，适合 LoRA/全参 SFT。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: object,
        configuration: Qwen3VLPolicyConfiguration,
    ):
        self.model = model
        self.processor = processor
        self.configuration = configuration

    @property
    def device(self) -> torch.device:
        """返回模型参数所在设备。"""
        return next(self.model.parameters()).device

    def _apply_chat(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        """把 chat 消息处理为模型输入张量。"""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in inputs.items()}

    @torch.inference_mode()
    def generate_actions(
        self,
        context: HistoryContext,
        include_past_actions: bool = True,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> tuple[list[StructuredAction], str]:
        """为一个上下文生成未来 ``action_horizon`` 帧的结构化动作。

        Parameters
        ----------
        context : HistoryContext
            多模态上下文。
        include_past_actions : bool
            是否在 prompt 中给出过去动作（实验对比"有历史/无历史"）。
        temperature : float
            0 表示贪心解码；>0 按温度采样，用于测生成自一致性。
        seed : int | None
            采样随机种子，None 表示不显式设定。

        Returns
        -------
        tuple[list[StructuredAction], str]
            解码后的定长动作块与模型原始文本。
        """
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
        context: HistoryContext,
        target_actions: list[StructuredAction],
    ) -> torch.Tensor:
        """对单个样本计算只监督目标动作 token 的交叉熵损失。

        prompt 部分与图像 token 的标签置为 -100，仅目标动作文本参与损失，
        构成标准的 VLA 行为克隆监督。返回标量 loss，可反传做 LoRA/全参 SFT。
        """
        if len(target_actions) != self.configuration.action_horizon:
            raise ValueError("target_actions 长度必须等于 action_horizon")
        messages = build_prompt_messages(
            context, self.configuration.action_format,
            self.configuration.action_horizon, include_past_actions=True,
        )
        prompt_inputs = self._apply_chat(messages, add_generation_prompt=True)
        target_text = encode_actions(target_actions, self.configuration.action_format)
        target_ids = self.processor.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(self.device)
        input_ids = torch.cat([prompt_inputs["input_ids"], target_ids], dim=1)
        attention = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
        model_inputs = {
            name: value for name, value in prompt_inputs.items()
            if name not in ("input_ids", "attention_mask")
        }
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            **model_inputs,
        )
        return outputs.loss


def build_qwen3vl_action_policy(
    configuration: Qwen3VLPolicyConfiguration,
    device: torch.device,
    cache_directory: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    lora_rank: int = 0,
) -> Qwen3VLActionPolicy:
    """加载 Qwen3VL 权重与 processor 并构造动作策略。

    Parameters
    ----------
    configuration : Qwen3VLPolicyConfiguration
        策略配置。
    device : torch.device
        模型放置设备。
    cache_directory : str | None
        Hugging Face 权重缓存目录。
    torch_dtype : torch.dtype
        模型计算精度，默认 bfloat16。
    lora_rank : int
        >0 时用 PEFT 给语言塔注意力/MLP 注入 LoRA 适配器，仅训练适配器；
        0 表示不加 LoRA（推理或全参）。

    Returns
    -------
    Qwen3VLActionPolicy
        就绪的动作策略。
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        configuration.model_name,
        cache_dir=cache_directory,
        max_pixels=configuration.image_max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        configuration.model_name,
        dtype=torch_dtype,
        cache_dir=cache_directory,
    )
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model = model.to(device)
    return Qwen3VLActionPolicy(model, processor, configuration)
