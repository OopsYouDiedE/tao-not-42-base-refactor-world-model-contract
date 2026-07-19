# Godot 训练链路方法说明

## 调用链

    train_ppo.main
      ├─ launch_godot
      ├─ GodotVectorizedEnvironment
      │    └─ GodotTrainingEnvironment
      │         ├─ wait_obs
      │         ├─ read_images / read_meta
      │         └─ send_action
      └─ stable_baselines3.PPO.learn

Godot 侧 TrainingCoordinator.cs 创建 40 个环境并使用文件后端 MemoryMappedFile 发布观测。
Python 侧 shared_memory_environment.py 打开同一个文件。

## 锁步协议

1. Godot 写入图像与元数据，然后递增 ObsSeq。
2. Python 的 wait_obs() 发现新序号并读取观测。
3. Python 写入连续与离散动作。
4. Python 将 ActSeq 写成已消费的 ObsSeq。
5. Godot 只在 ActSeq == ObsSeq 后推进下一步。

该协议保证每次动作对应一个已读取观测。共享内存布局的 Shape、Dtype 与偏移见
[README.md](README.md)。

## 训练装配

GodotVectorizedEnvironment 将共享内存协议适配为 SB3 VecEnv：

- 图像：[40, 128, 128, 3]，uint8。
- 时间：[40, 1]，float32，单位为秒。
- 动作：[40, 4]，int32/布尔语义，对应上、下、左、右。

train_ppo.main 负责启动 Godot、构造环境、训练与保存 checkpoint。训练结束时
关闭环境并终止由该进程启动的 Godot 子进程。
