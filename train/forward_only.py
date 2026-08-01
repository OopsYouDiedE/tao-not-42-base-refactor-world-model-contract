"""只跳过反向传播的 Hugging Face Trainer 训练步。"""

from __future__ import annotations

from typing import Any


class SkipBackwardTrainerMixin:
    """复用父类训练步，仅把 accelerator.backward 替换为计数空操作。"""

    backward_calls_skipped: int = 0

    def training_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: Any = None,
    ) -> Any:
        original_backward = self.accelerator.backward

        def skip_backward(*args: Any, **kwargs: Any) -> None:
            self.backward_calls_skipped += 1

        self.accelerator.backward = skip_backward
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            self.accelerator.backward = original_backward
