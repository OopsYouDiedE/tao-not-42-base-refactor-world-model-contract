# 真实窗口 10xx #1370 动作标注（看图核验版）

基于 `mosaic.png` / `step0-7.png` 的**直接视觉观察** + `actions.json` 真值动作产出。
场景：户外、暗绿色地面（夜晚或暗色生物群系的草地），地上摆着浅色**橡木木板**（同心方形
木纹）排成加号/菱形，屏幕底部为热键栏 HUD。任务：`obtain a diamond pickaxe`。
去抖阈值 `|Δbin| ≤ 1`（相机分量绝对值 ≤1 视为抖动，不出标注）。

---

**Step 0 — 真值 `use cam(+1,-1)`**（相机 ±1 抖动，已滤）
```
Scene: 俯视地面，3-4 块橡木木板排成 L/角形，中心一块颜色偏亮（刚放下）。
Type: mouse
Key: right_click
Action: press
Explanation: 准星对准地面木板阵列的空位，use（右键）用于放置方块，正在把橡木木板补进图案。
```

**Step 1 — 真值 `F jump cam(-2,+3)`**
```
Scene: 木板簇位于画面左侧，视角边前进边下压，地面木纹清晰。
Type: keyboard
Key: w
Action: hold
Explanation: 前进（w）靠近木板阵列继续铺设；跳跃可能用于越过已放方块或调整站位。
Type: keyboard
Key: space
Action: press
Explanation: 跳跃配合前进，避免卡在已放置的木板边缘。
Type: mouse
Key: camera
Action: move
Explanation: 视角左转并下压（yaw-2/pitch+3），把镜头对准左前方地面的下一个放置点。
```

**Step 2 — 真值 `F jump cam(+0,+1)`**（相机 pitch+1 抖动，已滤）
```
Scene: 木板加号图案居中，视角稳定俯视。
Type: keyboard
Key: w
Action: hold
Explanation: 持续前进，保持在木板阵列附近作业。
Type: keyboard
Key: space
Action: press
Explanation: 继续跳跃移动，微调与方块的相对高度/位置。
```

**Step 3 — 真值 `F jump cam(+2,+2)`**
```
Scene: 木板簇偏画面右上，视角右转下压。
Type: keyboard
Key: w
Action: hold
Explanation: 前进接近阵列右侧。
Type: keyboard
Key: space
Action: press
Explanation: 跳跃辅助定位。
Type: mouse
Key: camera
Action: move
Explanation: 视角右转下压（yaw+2/pitch+2），转向阵列右侧的空位。
```

**Step 4 — 真值 `cam(+2,+3)`**
```
Scene: 木板簇在画面左侧，右下角露出一块单独方块，视角明显右转下压。
Type: mouse
Key: camera
Action: move
Explanation: 仅调整视角（右转+低头），环视地面确认阵列布局与下一步放置位置，无移动或交互键。
```

**Step 5 — 真值 `B cam(+3,+3)`**
```
Scene: 后退使木板加号图案完整进入画面，视角右转下压。
Type: keyboard
Key: s
Action: hold
Explanation: 后退（s）拉开与阵列的距离，把整个加号图案纳入视野以便检查/继续扩建。
Type: mouse
Key: camera
Action: move
Explanation: 视角右转下压，配合后退保持镜头对准地面图案。
```

**Step 6 — 真值 `B R h2 cam(+1,+0)`**（相机 yaw+1 抖动，已滤）
```
Scene: 继续后退，木板阵列偏左，右下有独立方块，热键栏高亮格切换。
Type: keyboard
Key: s
Action: hold
Explanation: 后退移动到新的作业位置。
Type: keyboard
Key: d
Action: hold
Explanation: 右移（d）横向对齐到阵列右侧，配合后退绕到下一个放置点。
Type: keyboard
Key: 2
Action: press
Explanation: 切换热键栏第 2 格更换手持物品，对应底部 HUD 高亮格变化。
```

**Step 7 — 真值 `B R h1 cam(+2,+1)`**（相机 pitch+1 抖动，已滤）
```
Scene: 木板阵列居中偏上，右下独立方块仍在，热键栏切回第 1 格。
Type: keyboard
Key: s
Action: hold
Explanation: 继续后退调整站位。
Type: keyboard
Key: d
Action: hold
Explanation: 右移微调与阵列的相对位置。
Type: keyboard
Key: 1
Action: press
Explanation: 切回热键栏第 1 格换回主用方块/工具，准备下一次放置。
Type: mouse
Key: camera
Action: move
Explanation: 视角右转（yaw+2），把镜头带向下一个目标区域。
```
