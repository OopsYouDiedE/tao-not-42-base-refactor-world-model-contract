"""验证 Godot/Python 共享内存协议的布局与握手。

对外接口：test_shared_memory_round_trip（协议回环测试）。
"""

import mmap
import struct

import numpy as np

from rl_training_environments.godot import shared_memory_environment as environment_contract


def test_shared_memory_round_trip(tmp_path, monkeypatch):
    """验证观测读取和动作应答。

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest 临时目录，不含张量。
    monkeypatch : pytest.MonkeyPatch
        pytest 环境变量替换器，不含张量。

    Returns
    -------
    None
        测试通过时无返回值。
    """
    path = tmp_path / "godot.bin"
    path.write_bytes(b"\0" * environment_contract.TOTAL_SHM_SIZE)
    monkeypatch.setenv("RL_SHM_PATH", str(path))

    environment = environment_contract.GodotTrainingEnvironment(
        connect_timeout_s=0.1, poll_sleep_s=0.0,
    )
    try:
        environment.shm[0:environment_contract.TOTAL_IMAGES_BYTES] = (
            bytes([7]) * environment_contract.TOTAL_IMAGES_BYTES
        )
        metadata = np.arange(
            environment_contract.NUM_ENVS * environment_contract.META_PER_ENV,
            dtype=np.float32,
        )
        environment.shm[
            environment_contract.META_OFFSET:
            environment_contract.META_OFFSET + environment_contract.TOTAL_META_BYTES
        ] = metadata.tobytes()
        struct.pack_into("<i", environment.shm, environment_contract.OBS_SEQ_OFFSET, 1)

        assert environment.wait_for_observation(timeout_ms=10)
        assert environment.read_image_observations().shape == (
            environment_contract.NUM_ENVS,
            environment_contract.IMAGE_HEIGHT,
            environment_contract.IMAGE_WIDTH,
            environment_contract.CHANNELS,
        )
        assert environment.read_image_observations().dtype == np.uint8
        assert environment.read_image_observations()[0, 0, 0, 0] == 7
        assert np.array_equal(environment.read_metadata().reshape(-1), metadata)

        environment.send_actions(
            np.zeros((environment_contract.NUM_ENVS, environment_contract.CONT_DIM), dtype=np.float32),
            np.zeros((environment_contract.NUM_ENVS, environment_contract.DISC_DIM), dtype=np.int32),
        )
        assert struct.unpack_from(
            "<i", environment.shm, environment_contract.ACT_SEQ_OFFSET,
        )[0] == 1
    finally:
        environment.close()

    with path.open("r+b") as handle:
        mapped = mmap.mmap(
            handle.fileno(), environment_contract.TOTAL_SHM_SIZE, access=mmap.ACCESS_READ,
        )
        try:
            assert struct.unpack_from(
                "<i", mapped, environment_contract.ACT_SEQ_OFFSET,
            )[0] == 1
        finally:
            mapped.close()
