"""SB3 PPO 模型构造工厂。

对外接口：PPO_HYPERPARAMETERS、build_ppo_model。

只放与具体任务无关的"怎么造"逻辑；具体采集/学习循环属于各 train/godot_meta_rl/*。
"""

from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger

# 三种执行器（锁步 / 线程异步 / 双进程）共用同一组 PPO 超参，保证"只变执行方式"这一个变量。
PPO_HYPERPARAMETERS = dict(n_steps=64, batch_size=256, n_epochs=4,
                           gamma=0.99, gae_lambda=0.95, ent_coef=0.01)


def build_ppo_model(
    vectorized_environment,
    device="auto",
    verbose=0,
    with_null_logger=True,
    **overrides,
):
    """用统一超参构造 PPO(MultiInputPolicy)。

    Parameters
    ----------
    vectorized_environment : VecEnv
        已套好 VecMonitor/VecFrameStack 的向量环境。
    with_null_logger : bool  True 时挂一个空 logger——不走 .learn() 而直接调 PPO.train() 的执行器需要它。
    overrides : dict         覆盖或补充 PPO_HYPERPARAMETERS。
    """
    hyperparameters = dict(PPO_HYPERPARAMETERS)
    hyperparameters.update(overrides)
    model = PPO(
        "MultiInputPolicy", vectorized_environment,
        device=device, verbose=verbose, **hyperparameters,
    )
    if with_null_logger:
        model.set_logger(Logger(folder=None, output_formats=[]))
    return model
