#!/usr/bin/env python3
"""合成技能:脚本化 GUI 操作在 Craftground V2 里合成(已验证 oak_log→oak_planks)。

破解的 GUI 点击机制(craftground mod MouseInfo.kt / MinecraftEnv.kt:464-477):
  - attack=GLFW 左键、use=右键、sneak=左Shift(modifier);
  - camera_yaw/pitch × (20/3)=6.67 → 光标像素位移(moveMouseBy,逐像素触发 hover);
  - **点击必须与 camera 移动同帧**(applyAction 里 moveMouseBy 先于 onAction),否则光标不在槽上;
  - **sneak+attack 点输出槽 = shift-click**:自动合成全部并直送背包(绕开逐格存取的光标漂移)。
  - 开背包时 MouseInfo 光标 = 屏幕中心 (320,180)。

槽位像素(640×360 居中背包):inventory.0 主格左上≈(247,198);合成 2×2 左上≈(343,123);输出≈(419,124)。

用法(ZEROCOPY :1):
  DISPLAY=:1 PYTHONPATH=. ./.venv/bin/python tests/integration/craft_skill.py
"""

DEG = 20.0 / 3                         # camera 1° = 6.67 光标像素(mod 常数)
SLOT_INV0 = (247, 198)                 # 主背包左上(item replace inventory.0 落点)
SLOT_GRID = (343, 123)                 # 合成 2×2 左上格
SLOT_OUT = (419, 124)                  # 合成输出槽


class GuiCursor:
    """背包 GUI 光标控制。开背包时光标在屏幕中心;click_at 移动到槽并同帧点击(拖拽点击)。"""

    def __init__(self, env, noop, step_fn):
        self.env, self.noop, self.step = env, noop, step_fn
        self.cur = [320, 180]

    def click_at(self, tx, ty, button, n=8, hold=None):
        """移到 (tx,ty) 并点击。button=attack(左)/use(右);hold=sneak → shift-click。"""
        dx, dy = tx - self.cur[0], ty - self.cur[1]
        for i in range(n):
            a = dict(self.noop, camera_yaw=(dx / DEG) / n, camera_pitch=(dy / DEG) / n)
            if hold:
                a[hold] = True
            if i == n - 1:                # 末帧(到位)带按键 = 点击落在槽上
                a[button] = True
            self.step(a)
        self.step()                       # 释放
        self.cur = [tx, ty]


# 合成 2×2 四格 + 输出槽附近扫掠点(覆盖 ±像素漂移,均幂等)
GRID_SLOTS = [(343, 123), (363, 123), (343, 143), (363, 143)]
OUT_SWEEP = [(419, 124), (423, 121), (415, 128), (419, 128)]


def craft_from_grid(cur: GuiCursor, src_slot):
    """抓 src_slot 整叠材料 → 左键扫合成格(首格落整叠)→ sneak+attack 扫输出(自动合成全部到背包)。

    仅适用无需精确布局的配方(如 1×log→4planks:整叠留格,shift-click 重复合成耗尽)。抗光标漂移。
    """
    cur.click_at(*src_slot, "attack")                 # 抓整叠材料 → 光标
    for gx, gy in GRID_SLOTS:                          # 左键扫格:首个命中格落整叠,余下空点无效
        cur.click_at(gx, gy, "attack")
    for ox, oy in OUT_SWEEP:                           # shift-click 输出:自动合成全部直送背包
        cur.click_at(ox, oy, "attack", hold="sneak")


def main():
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    import os
    zc = os.environ.get("DISPLAY", "") == ":1"
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.ZEROCOPY if zc else ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="craft",
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=9250, verbose=False)
    noop = no_op_v2()
    obs = {}

    def step(a=None):
        nonlocal obs
        obs = env.step(a or dict(noop))[0]
        return obs

    def inv():
        return {it.translation_key.split(".")[-1]: it.count
                for it in obs["full"].inventory if it.count > 0}

    env.reset(options={"fast_reset": True, "extra_commands": ["clear @p"]})
    for _ in range(3):
        step()
    env.reset(options={"fast_reset": True,
                       "extra_commands": ["item replace entity @p inventory.0 with minecraft:oak_log 5"]})
    for _ in range(12):
        step()
    step(dict(noop, inventory=True)); step(); step()
    print("[craft] 开局:", inv(), flush=True)
    craft_from_grid(GuiCursor(env, noop, step), SLOT_INV0)
    step(dict(noop, inventory=True))
    for _ in range(6):
        step()
    final = inv()
    ok = final.get("oak_planks", 0) > 0
    print(f"[craft] 最终库存: {final}  →  {'PASS 合成成立' if ok else 'FAIL'}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
