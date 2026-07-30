from datasets.action_codec import decode_lumine_action
from tools.craftground_closed_loop_server import _chunk_action


def test_project_action_chunk_maps_to_craftground_v2() -> None:
    chunk = decode_lumine_action(
        "<|action_start|> ; Mouse -20 10 W A MouseRight <|action_end|>"
    ).chunks[0]
    action = _chunk_action(chunk.keys, chunk.mouse)
    assert action["forward"] is True
    assert action["left"] is True
    assert action["use"] is True
    assert action["camera_yaw"] == -3.0
    assert action["camera_pitch"] == 1.5
    assert action["attack"] is False


def test_absent_key_releases_on_next_tick() -> None:
    chunks = decode_lumine_action(
        "<|action_start|> ; W ; Mouse 2 -1 <|action_end|>"
    ).chunks
    first = _chunk_action(chunks[0].keys, chunks[0].mouse)
    second = _chunk_action(chunks[1].keys, chunks[1].mouse)
    assert first["forward"] is True
    assert second["forward"] is False
    assert second["camera_yaw"] == 0.3
    assert second["camera_pitch"] == -0.15
