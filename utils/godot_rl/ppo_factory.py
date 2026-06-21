"""SB3 PPO 的可复用工厂与 rollout buffer 小数组搬运助手（从各异步训练脚本抽出的工厂函数）。

对外接口：PPO_HP（统一超参）、build_model（构造 PPO）、make_buffer（构造 DictRolloutBuffer）、
small_keys / extract_small / bind_small（把 buffer 里除图像外的小数组取出/绑回，配合图像走共享内存零拷贝）。

只放与具体任务无关的"怎么造"逻辑；具体采集/学习循环属于各 train/godot_meta_rl/*。
"""

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import Logger

# 三种执行器（锁步 / 线程异步 / 双进程）共用同一组 PPO 超参，保证"只变执行方式"这一个变量。
PPO_HP = dict(n_steps=64, batch_size=256, n_epochs=4,
              gamma=0.99, gae_lambda=0.95, ent_coef=0.01)


def build_model(venv, device="auto", verbose=0, with_null_logger=True, **overrides):
    """用统一超参构造 PPO(MultiInputPolicy)。

    Parameters
    ----------
    venv : VecEnv            已套好 VecMonitor/VecFrameStack 的向量环境。
    with_null_logger : bool  True 时挂一个空 logger——不走 .learn() 而直接调 PPO.train() 的执行器需要它。
    overrides : dict         覆盖/补充 PPO_HP 的超参。
    """
    hp = dict(PPO_HP)
    hp.update(overrides)
    model = PPO("MultiInputPolicy", venv, device=device, verbose=verbose, **hp)
    if with_null_logger:
        model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def make_buffer(model, obs_space=None, act_space=None):
    """按 model 的形状/超参构造一个 DictRolloutBuffer（双缓冲场景每份各造一个）。"""
    return DictRolloutBuffer(
        model.n_steps,
        obs_space if obs_space is not None else model.observation_space,
        act_space if act_space is not None else model.action_space,
        device=model.device, gae_lambda=model.gae_lambda,
        gamma=model.gamma, n_envs=model.n_envs,
    )


def small_keys():
    """rollout buffer 里除图像外需跨进程搬运的小数组键（图像太大走共享内存零拷贝，不在此列）。"""
    return ["sim_dt", "actions", "values", "log_probs", "advantages",
            "returns", "rewards", "episode_starts"]


def extract_small(buf):
    """从一个填满+算好 GAE 的 buffer 取出除图像外的小数组（拷贝，走 Queue）。"""
    return {
        "sim_dt": buf.observations["sim_dt"].copy(),
        "actions": buf.actions.copy(),
        "values": buf.values.copy(),
        "log_probs": buf.log_probs.copy(),
        "advantages": buf.advantages.copy(),
        "returns": buf.returns.copy(),
        "rewards": buf.rewards.copy(),
        "episode_starts": buf.episode_starts.copy(),
    }


def bind_small(buf, img_view, small):
    """把一个 buffer 的内部数组指向给定图像视图 + 小数组，供 PPO.train 直接读（零额外拷贝）。"""
    buf.observations["image"] = img_view
    buf.observations["sim_dt"] = small["sim_dt"]
    buf.actions = small["actions"]
    buf.values = small["values"]
    buf.log_probs = small["log_probs"]
    buf.advantages = small["advantages"]
    buf.returns = small["returns"]
    buf.rewards = small["rewards"]
    buf.episode_starts = small["episode_starts"]
