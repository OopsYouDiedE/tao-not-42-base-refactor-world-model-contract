from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from train.forward_only import SkipBackwardTrainerMixin


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.tensor([float(index), 1.0])
        return {"input": value, "labels": value.sum().reshape(1)}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 1)
        self.forward_calls = 0

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        prediction = self.projection(input)
        return {"loss": nn.functional.mse_loss(prediction, labels)}


class ForwardOnlyTrainer(SkipBackwardTrainerMixin, Trainer):
    pass


def test_forward_only_runs_real_training_loop_without_parameter_updates(tmp_path) -> None:
    model = TinyModel()
    initial = copy.deepcopy(model.state_dict())
    trainer = ForwardOnlyTrainer(
        model=model,
        train_dataset=TinyDataset(),
        args=TrainingArguments(
            output_dir=tmp_path,
            max_steps=2,
            per_device_train_batch_size=1,
            save_strategy="steps",
            save_steps=1,
            logging_steps=1,
            report_to="none",
        ),
    )

    result = trainer.train()

    assert result.global_step == 2
    assert trainer.backward_calls_skipped == 2
    assert model.forward_calls == 2
    assert result.training_loss > 0
    assert list(tmp_path.glob("checkpoint-*"))
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name])
