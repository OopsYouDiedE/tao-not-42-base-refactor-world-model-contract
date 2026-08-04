from pathlib import Path

from behavior_cloning_dataset_converters.utils import build_grouped_split, load_split


def test_build_grouped_split_accepts_dataset_independent_episode_names(tmp_path: Path) -> None:
    output_path = tmp_path / "split.json"

    result = build_grouped_split(
        episode_frames={"alpha": 80, "beta": 40, "gamma": 20},
        episode_groups={"alpha": "source-a", "beta": "source-b", "gamma": "source-b"},
        validation_ratio=0.4,
        output_path=output_path,
    )

    assert result.validation_episodes == ["beta", "gamma"]
    assert result.validation_groups == ["source-b"]
    assert load_split(output_path) == result


def test_episode_holdout_does_not_require_group_mapping() -> None:
    result = build_grouped_split(
        episode_frames={"plain-name-a": 10, "plain-name-b": 10, "plain-name-c": 10},
        holdout_level="episode",
        validation_ratio=0.3,
        seed=7,
    )

    assert len(result.validation_episodes) == 1
    assert len(result.train_episodes) == 2
