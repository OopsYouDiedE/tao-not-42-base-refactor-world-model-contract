"""验证 Godot/Python 文件映射协议的布局与握手。

对外接口：test_shared_memory_round_trip（协议回环测试）。
"""

import mmap
import struct

import numpy as np

from utils.godot_rl import shared_mem_env as E


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
    path.write_bytes(b"\0" * E.TOTAL_SHM_SIZE)
    monkeypatch.setenv("RL_SHM_PATH", str(path))

    env = E.GodotTrainEnv(connect_timeout_s=0.1, poll_sleep_s=0.0)
    try:
        env.shm[0:E.TOTAL_IMAGES_BYTES] = bytes([7]) * E.TOTAL_IMAGES_BYTES
        meta = np.arange(E.NUM_ENVS * E.META_PER_ENV, dtype=np.float32)
        env.shm[E.META_OFFSET:E.META_OFFSET + E.TOTAL_META_BYTES] = meta.tobytes()
        struct.pack_into("<i", env.shm, E.OBS_SEQ_OFFSET, 1)

        assert env.wait_obs(timeout_ms=10)
        assert env.read_images().shape == (E.NUM_ENVS, E.IMAGE_HEIGHT, E.IMAGE_WIDTH, E.CHANNELS)
        assert env.read_images().dtype == np.uint8
        assert env.read_images()[0, 0, 0, 0] == 7
        assert np.array_equal(env.read_meta().reshape(-1), meta)

        env.send_action(
            np.zeros((E.NUM_ENVS, E.CONT_DIM), dtype=np.float32),
            np.zeros((E.NUM_ENVS, E.DISC_DIM), dtype=np.int32),
        )
        assert struct.unpack_from("<i", env.shm, E.ACT_SEQ_OFFSET)[0] == 1
    finally:
        env.close()

    with path.open("r+b") as handle:
        mapped = mmap.mmap(handle.fileno(), E.TOTAL_SHM_SIZE, access=mmap.ACCESS_READ)
        try:
            assert struct.unpack_from("<i", mapped, E.ACT_SEQ_OFFSET)[0] == 1
        finally:
            mapped.close()
