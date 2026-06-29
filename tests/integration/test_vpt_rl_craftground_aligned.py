#!/usr/bin/env python3
"""测试 VPT RL fine-tuned 模型在 CraftGround 上的表现 (使用 V2 键鼠动作空间直接对齐)"""
import pickle
import torch as th
import numpy as np
from train.craftground.env import MinecraftCraftgroundEnv
from net.vpt_lib.policy import MinecraftAgentPolicy
from net.vpt_lib.action_mapping import CameraHierarchicalMapping
from net.vpt_lib.actions import ActionTransformer
from net.vpt_lib.gym3_types import DictType
from craftground.environment.action_space import no_op_v2

def to_craftground_v2(buttons, camera):
    """将 MineRL 的按钮状态和相机变动，直接映射到 CraftGround 的 V2 键鼠动作空间字典"""
    cg_action = no_op_v2()
    
    # 填充布尔按键 (包含新增加的 drop 和 inventory)
    for k in ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop", "inventory"]:
        if k in buttons:
            cg_action[k] = bool(buttons[k])
            
    # 填充快捷栏 (支持 no_op_v2 的 dot 键和文档推荐的 under 键)
    for i in range(1, 10):
        k_dot = f"hotbar.{i}"
        k_under = f"hotbar_{i}"
        
        # MineRL 原生使用的是 "hotbar.1" 等，我们双填以确保完美兼容
        val = False
        if k_dot in buttons:
            val = bool(buttons[k_dot])
        elif k_under in buttons:
            val = bool(buttons[k_under])
            
        cg_action[k_dot] = val
        cg_action[k_under] = val
    # 填充连续相机 pitch 和 yaw
    pitch, yaw = camera
    cg_action["camera_pitch"] = float(pitch)
    cg_action["camera_yaw"] = float(yaw)
    
    return cg_action

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", default=":1", help="DISPLAY target (e.g. :1 for headless GPU Xorg)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments to run")
    parser.add_argument("--max-steps", type=int, default=36000, help="Maximum steps to run per env")
    args = parser.parse_args()
    
    # 优先将 DISPLAY 绑定到无头 GPU 渲染器 (若指定且非空)
    if args.display:
        os.environ['DISPLAY'] = args.display
    print(f"=== VPT RL Fine-tuned 模型测试 (直接对齐 V2 键鼠动作空间, DISPLAY={os.environ.get('DISPLAY')}) ===")
    
    # 1. 载入配置与初始化完整的 Agent Policy
    model_path = 'runs/checkpoints/vpt/weights/2x.model'
    weights_path = 'runs/checkpoints/vpt/weights/rl-from-early-game-2x.weights'
    
    print("载入配置并初始化 Policy...")
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    policy_kwargs = params['model']['args']['net']['args']
    pi_head_kwargs = params['model']['args']['pi_head_opts']
    pi_head_kwargs['temperature'] = float(pi_head_kwargs['temperature'])

    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = action_mapper.get_action_space_update()
    action_space = DictType(**action_space)

    policy = MinecraftAgentPolicy(action_space=action_space, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs).cuda()
    
    print("加载官方预训练权重 (包含 pi_head)...")
    policy.load_state_dict(th.load(weights_path, map_location='cuda'), strict=False)
    policy.eval()
    print("✓ 权重加载成功")

    action_transformer = ActionTransformer(
        camera_binsize=2,
        camera_maxval=10,
        camera_mu=10,
        camera_quantization_scheme="mu_law"
    )

    # 2. 创建环境 (根据 DISPLAY 自动选择最佳编码模式)
    n_envs = args.n_envs
    max_steps = args.max_steps
    from craftground.screen_encoding_modes import ScreenEncodingMode
    
    current_display = os.environ.get("DISPLAY", "")
    if current_display == ":1":
        enc_mode = ScreenEncodingMode.ZEROCOPY
        enc_desc = "ZEROCOPY GPU 零拷贝模式"
    else:
        enc_mode = ScreenEncodingMode.RAW
        enc_desc = "RAW 像素回传模式"
        
    print(f"创建 {n_envs} 个并行环境（{max_steps} 步/episode，使用 {enc_desc}，DISPLAY={current_display}）...")

    from train.craftground.env import MinecraftCraftgroundEnv
    envs = [
        MinecraftCraftgroundEnv(seed=100+i, max_steps=max_steps, port=9100+i*5, screen_encoding_mode=enc_mode)
        for i in range(n_envs)
    ]

    episodes_rewards_history = []
    episodes_lens_history = []

    running_rewards = np.zeros(n_envs)
    obs = [env.reset() for env in envs]
    state = policy.initial_state(n_envs)
    state = [s.cuda() if isinstance(s, th.Tensor) else s for s in state]
    first = th.from_numpy(np.array([True] * n_envs)).cuda()

    for step in range(max_steps):
        if isinstance(obs[0], th.Tensor):
            obs_t = th.stack(obs)
        else:
            obs_t = th.from_numpy(np.array(obs)).cuda()  # [B, H, W, C]
        obs_t = obs_t.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        # 直接使用 GPU 批处理重采样，杜绝循环和 CPU 拷贝
        obs_t = th.nn.functional.interpolate(
            obs_t, size=(128, 128), mode="bilinear", align_corners=False
        )
        obs_t = obs_t.permute(0, 2, 3, 1)  # 恢复到 [B, H, W, C] 保持 0-255，并在 CUDA 上

        with th.no_grad():
            agent_input = {"img": obs_t}
            agent_action, state, _ = policy.act(
                agent_input, first, state, stochastic=True
            )

        # 解包并反解
        action = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera": agent_action["camera"].cpu().numpy()
        }
        minerl_action = action_mapper.to_factored(action)
        minerl_action_transformed = action_transformer.policy2env(minerl_action)

        dones = np.zeros(n_envs, dtype=bool)
        next_obs = []

        for i in range(n_envs):
            buttons_i = {}
            for k, v in minerl_action_transformed.items():
                if k == 'camera':
                    continue
                if isinstance(v, np.ndarray):
                    buttons_i[k] = bool(v[i])
                else:
                    buttons_i[k] = bool(v)

            camera_arr = minerl_action_transformed.get('camera', np.array([[0.0, 0.0]]))
            camera_i = camera_arr[i] if camera_arr.ndim > 1 else camera_arr
            camera_i = (float(camera_i[0]), float(camera_i[1]))

            # 直接通过底层的 CraftGroundEnvironment 执行 V2 键鼠动作 dict
            cg_v2_action = to_craftground_v2(buttons_i, camera_i)
            result = envs[i].env.step(cg_v2_action)

            if len(result) == 5:
                obs_dict, _rew, done, truncated, _info = result
                done = done or truncated
            else:
                obs_dict, _rew, done, _info = result

            # 手动更新 env 的计数和内在奖励计算
            envs[i].episode_step += 1
            full_obs = obs_dict.get("full", None)
            intrinsic = 0.0
            if full_obs is not None:
                intrinsic, new_ach_indices, force_done = envs[i].reward_shaper.compute(
                    full_obs, envs[i].episode_step
                )
                
                # 打印新解锁的成就
                if new_ach_indices:
                    from train.craftground.achievements import ALL_ACHIEVEMENTS
                    for idx in new_ach_indices:
                        print(f"    [Env {i}] ⭐ 解锁成就: {ALL_ACHIEVEMENTS[idx]}")
                        
                if force_done:
                    print(f"    [Env {i}] ⚠️ 触发强制重置 (force_done=True)，内在处罚后本步奖励: {intrinsic:.2f}")
                    done = True

            running_rewards[i] += intrinsic
            dones[i] = done

            if done or envs[i].episode_step >= envs[i].max_steps:
                print(f"✓ [Env {i}] Episode finished: reward={running_rewards[i]:.2f}, len={envs[i].episode_step}")
                print(f"  🏆 本集已解锁成就: {list(envs[i].reward_shaper.unlocked)}")
                print(f"  🎒 本集收集到的物品类型: {list(envs[i].reward_shaper.seen_item_keys)}")
                
                episodes_rewards_history.append(running_rewards[i])
                episodes_lens_history.append(envs[i].episode_step)

                obs_i = envs[i].reset()
                running_rewards[i] = 0.0
            else:
                obs_i = obs_dict["rgb"]

            next_obs.append(obs_i)

        obs = next_obs
        first = th.from_numpy(dones).cuda()

        if (step + 1) % 500 == 0:
            print(f"步数 {step+1}/{max_steps}: 已完成 {len(episodes_rewards_history)} 个 Episodes")

    for env in envs:
        env.close()

    print(f"\n=== 结果 ===")
    if episodes_rewards_history:
        print(f"平均奖励: {np.mean(episodes_rewards_history):.2f} ± {np.std(episodes_rewards_history):.2f}")
        print(f"平均长度: {np.mean(episodes_lens_history):.1f}")
        print(f"奖励范围: [{np.min(episodes_rewards_history):.2f}, {np.max(episodes_rewards_history):.2f}]")
    else:
        print("无完成的 Episode。")
