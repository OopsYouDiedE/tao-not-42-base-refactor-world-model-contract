# 两套 Minecraft 渲染器 — WSL 从零复现文档

本文档覆盖两条独立的渲染路，均在 **WSL2 Ubuntu 24.04**（本机有 RTX 3070）上 2026-07-24 实证跑通：

- **第一部分：solaris 渲染引擎 × mineflayer**（`prismarine-viewer-colalab` = three.js +
  node-canvas-webgl + headless-gl），渲染真实连到 MC 1.21 server 的 mineflayer bot。
  验收 = 每个动作起止时截图 + 全程录像。
- **第二部分：CraftGround 渲染器**（Minecraft Java 版 + Fabric mod，LWJGL/OpenGL）。
  验收 = 每秒截图 + 全程录像。见文末「第二部分」。

> solaris 部分:只覆盖 headless 渲染路(绕开需要 NVENC GPU 的 camera 路)。所有步骤除"装 -dev 库"外均免 sudo。

**脚本位置**:本文档引用的 `_*.sh` / `_*.js` 脚本都在
`C:\Users\iii\Desktop\mc-agents\solaris-engine\`(WSL 路径 `/mnt/c/Users/iii/Desktop/mc-agents/solaris-engine/`)。
mc-agents 是独立于本世界模型项目的环境(勿嵌套),故脚本留在那边、文档留在本项目。

---

## 0. 环境事实(本机实测)

- WSL2 Ubuntu 24.04,node v22.22.1,npm 10.9.4,gcc 13.3。
- **工作目录 `~/mc-test/`**(WSL 原生 fs,不要放 `/mnt/c`——原生 fs 编译快很多)。
- npm 走本地代理 `127.0.0.1:7897`,registry 阿里云镜像;github/codeload/npmjs 均可达。
- **sudo 需要密码**(只有装 -dev 库那一步用到,须人工执行)。
- 系统自带 java 是 1.8(跑不了 MC1.21),故用便携 JDK21,不动系统 java。

目录布局:
```
~/mc-test/
  solaris-engine/                 # 本仓库的拷贝(rsync,排除 node_modules)
  vendor/prismarine-viewer-colalab/  # 去掉 prepare 的 viewer fork(见步骤4)
  jdk21/                          # 便携 Temurin JDK21
  server/                         # PaperMC 1.21 + server.properties(离线+RCON)
  bin/python                      # python->python3 shim(headless-gl 编译要)
  output/                         # smoke_test 产物(mp4 + json)
  *.log                           # 各步骤日志
```

脚本执行约定(Windows Git Bash 下):
```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/iii/Desktop/mc-agents/solaris-engine/<脚本>.sh
```

---

## 1. 需要 sudo 的唯一一步:装 canvas 编译所需 -dev 库

canvas 原生编译需要头文件(runtime .so 本就在,缺的是 header + pkg-config .pc)。**在你的终端执行**:

```bash
!wsl -d Ubuntu -- sudo apt-get install -y \
  libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev \
  libpixman-1-dev pkg-config build-essential
```

验证:`_verify_dev.sh`  → pixman-1 / cairo / pangocairo 应全绿。

> 若无法 sudo:改走 solaris 自带 Dockerfile(已固化全部 -dev 库),但那是另一条路。

---

## 2. 拷贝仓库到 WSL 原生 fs + 装 python/cv2

```bash
mkdir -p ~/mc-test
rsync -a --exclude node_modules /mnt/c/Users/iii/Desktop/mc-agents/solaris-engine ~/mc-test/
# act_recorder 依赖(WSL 无 python3-venv 且 sudo 要密码,故 --user)
pip3 install --user --break-system-packages -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com "numpy<2" opencv-python-headless
```

---

## 3. MC 1.21 离线 server(便携 JDK21 + PaperMC)

脚本:**`_setup_server.sh`**(下 PaperMC 1.21 via v3 fill API + 写 server.properties:online-mode=false、enable-rcon、rcon.password=research、level-type=flat)。
JDK21 需先下:Temurin `OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz` 解压到 `~/mc-test/jdk21`。

```bash
_setup_server.sh      # 备好 jar + 配置
_start_server.sh      # 后台起 server,轮询直到 25565/25575 监听
_check_server.sh      # 验证 RCON 通(应返回 "0 of 20 players online")
```

> 注意:PaperMC 旧 v2 API 已 sunset,`_setup_server.sh` 用的是 v3 `fill.papermc.io`。

---

## 4. 装 node 依赖(核心 —— 绕开 webpack 回滚 + 原生编译)

**关键背景**:直接 `npm ci` 会失败并回滚。根因 = viewer fork 的 `prepare` 脚本
(`node viewer/prerender.js && webpack`)在打包浏览器端 viewer 时炸(`@ljharb/tsconfig` 解析,37 errors),
而 git-dep 的 prepare 失败会让整个 npm ci 回滚。headless 路根本不用 webpack 产物。

按顺序执行:

```bash
# 4a. 全局把 SSH github 重写成 HTTPS(治 git-dep 退回 ssh clone 的 Permission denied)
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
git config --global url."https://github.com/".insteadOf "git@github.com:"

# 4b. clone viewer fork 到 vendor/ 并删掉 prepare/prepublishOnly/prepack
_vendor_viewer.sh

# 4c. 把 solaris package.json 的 prismarine-viewer-colalab 改成 file:../vendor/...
#     然后 npm install(带 python shim + CXXFLAGS,见下)。_install5.sh 是最终有效版:
#       - PATH 前置 ~/mc-test/bin(python->python3 shim,headless-gl 的 ANGLE gyp 调 `python`)
#       - export CXXFLAGS="-include cstdint" CFLAGS="-include stdint.h"
#         (ANGLE 老代码在 gcc13 报 `uintptr_t does not name a type`)
_patch_and_install.sh   # 改 package.json 为 file: 依赖(会先跑一次 install)
_install5.sh            # 带 shim+CXXFLAGS 重装,产出 gl/canvas 的 .node
_wait_install5.sh       # 等待并验证 gl/build/Release/webgl.node + canvas.node

# 4d. vendored viewer 自身运行时依赖(three r128 等)+ 纹理源 minecraft-assets
_install_viewer_deps.sh # cd vendored && npm install(prepare 已删,无 webpack)
_prerender_setup.sh     # 读 prerender.js + 装 minecraft-assets

# 4e. file: 会变 symlink,node 按真实路径解析导致 viewer 找不到 hoist 的依赖。
#     改成实体复制进 node_modules(带上 three)
_desymlink_viewer.sh
```

> python shim 建法:`mkdir -p ~/mc-test/bin && ln -sf $(command -v python3) ~/mc-test/bin/python`,并 `export PATH="$HOME/mc-test/bin:$PATH"`。

---

## 5. 预生成渲染资源(否则渲染中途 ENOENT 崩)

```bash
# 5a. 生成 public/textures/<ver>.png 图集 + blocksStates/<ver>.json(含 1.21.4)
_run_prerender.sh       # 内部 node viewer/prerender.js -f

# 5b. 复刻 Dockerfile 37-40:玩家实体 skin 拷进 public/textures/1.16.4/entity/
#     (渲染对方玩家实体如 Bocchi.png 时需要)
_fix_skins.sh
```

验证整条渲染栈能加载并渲染一帧:
```bash
_run_probe.sh
# 期望:gl(stack-gl 6.0.2) / three(r128) / node-canvas-webgl / WebGLRenderer 渲染+JPEG / headless() 全 OK
```

---

## 6. 跑 smoke_test 单集

单 controller 会卡在 coordinator 等对端,故须起 **2 个 bot 互连** + 各配一个 act_recorder。
关键参数:`--enable_camera_wait 0`(跳过 NVENC camera)、`--smoke_test 1 --episodes_num 1`、
`--viewer_rendering_disabled 0`、`--world_type flat`。

```bash
_smoke.sh          # 起 2 act_recorder(8091/8092) + 2 controller(Alpha/Bravo, coord 8093<->8094)
sleep 90
_smoke_status.sh   # 看进度/日志/output
_verify_output.sh  # 硬指标:帧数、cv2 解码非黑、动作标注丰富度
```

**通过标准(实测值)**:每 bot ~320 帧 640×360 mp4(cv2 mean~150 std~18 nonzero100%,~18fps)
+ ~640KB json(逐帧 22 维动作契约 forward/back/camera[Δyaw,Δpitch]/attack/mine/hotbar.1-9 + inventory)。
controller 日志出 `All 1 episodes completed`、`encountered_error=false bot_died=false`。

> 本文档跑的默认集是 idle 场景(active_actions 空,仅 camera_moves)。要验按键/挖矿动作记录,换动作密集 episode 再跑。

---

## 附:脚本分类(在 `C:\Users\iii\Desktop\mc-agents\solaris-engine\`)

**最终有效路径**(按上文顺序):
`_verify_dev.sh` · `_setup_server.sh` · `_start_server.sh` · `_check_server.sh` ·
`_vendor_viewer.sh` · `_patch_and_install.sh` · `_install5.sh` · `_wait_install5.sh` ·
`_install_viewer_deps.sh` · `_prerender_setup.sh` · `_desymlink_viewer.sh` ·
`_run_prerender.sh` · `_fix_skins.sh` · `_run_probe.sh` · `_probe_render.js` ·
`_smoke.sh` · `_smoke_status.sh` · `_verify_output.sh`

**诊断/探路/失败迭代(可忽略或删除)**:
`_check_ncw.sh` `_check_status.sh` `_clean_restart_npm.sh` `_diag.sh` `_inspect_textures.sh`
`_install4.sh` `_install_split.sh` `_locate.sh` `_npmci_httpsonly.sh` `_snap.sh`
`_wait_ci2.sh` `_wait_install2.sh` `_wait_install3.sh` `_wait_install4.sh`
(install4/2/3 是加 shim/CXXFLAGS 之前的失败版,被 install5 取代)

---

## 7. 地形渲染修复(2026-07-25，关键)

初次跑通时用"cv2 帧非黑(mean>1)"当验收标准是**错的**——那只是天空色背景，地形根本没渲染，画面是玩家实体漂在灰蓝虚空里。**验收渲染要看 Canny 边缘密度 edges%：正常地形 9-13%，空画面 ~0.1%**。

根因是 vendored `prismarine-viewer-colalab` 对 Minecraft 1.18+ 负 y 世界（地面 y=-60，世界 -64..320）的多处旧假设，加一处版本错配。修复（改后各留 `.js.orig` 备份）：

1. **`viewer/lib/worker.js`**（2 处）：`chunk.sections[Math.floor(y / 16)]` → `chunk.sections[Math.floor((y - (chunk.minY ?? 0)) / 16)]`。现代 prismarine-chunk 的 sections 是从 `chunk.minY`(-64) 起 0-based 的数组，裸 `y/16` 对负 y 落负索引=undefined，地面 section 被当空跳过网格化。
2. **`viewer/lib/worldrenderer.js`**（2 处 addColumn/removeColumn）：`for (let y = 0; y < 256; y += 16)` → `for (let y = -64; y < 320; y += 16)`，否则负 y 地面 section 从不被标 dirty。
3. **`viewer/lib/models.js`**（2 处 cullface）：`if (neighbor.position.y < 0) continue` → `< -64`（低于世界底才当空气）。

4. **版本对齐（最后一块拼图）**：改完上面三处，地形能生成几何但呈**竖刺畸变**——根因是 server 是 MC **1.21.0**，而 viewer 的 `getVersion("1.21")` 归一到 **1.21.4**（supportedVersions 里无裸 "1.21"），用 1.21.4 的 section palette 解 1.21.0 chunk → 每方块画满六面（畸变时顶点数爆到 749 万）。**修法：三者版本完全对齐**——建 PaperMC **1.21.4** server（`_setup_server_1214.sh` / `_start_server_1214.sh`，端口 25567 / RCON 25577），bot 连时 `--mc_version 1.21.4`，viewer 也解码 1.21.4。对齐后顶点降到正常 ~12 万，渲染出正常 superflat 草地。（注意：强制 viewer 用 1.21.1 反而几乎全空——对齐必须往 1.21.4 走。）

验收：跑动作密集 mine episode（连 1.21.4 server）→ 出 mp4 + 逐帧动作 json → 本项目 `python -m rl_training_environments.solaris.acceptance_boundary_shots` 抽每个动作起止帧截图。实测出 8 个动作边界（hotbar 换镐 / camera 转头 / mine 挖矿的起止），全部真实草地画面。solaris engine controller 已 vendored 进 `rl_training_environments/solaris/engine/`，补丁在 `rl_training_environments/solaris/viewer_patches/`。

---

## 附:WSL 命令铁律(踩过多次)

- `wsl -d Ubuntu -- bash -lc '...'` 里内联 `$VAR` / `2>/dev/null` / `$(...)` / 嵌套引号,
  极易被外层 Git Bash 二次解析炸。**一律写成 .sh 脚本文件,再 `bash /mnt/c/.../x.sh` 执行**。
- 调用脚本时加 `MSYS_NO_PATHCONV=1`,防 `/mnt/c/...` 被 Git Bash 改写成 `C:/Program Files/Git/...`。
- 长任务(npm install / server 下载)用 `setsid ... < /dev/null & disown` 完全 detach,
  避免 CLI 120s 超时把进程一起截断(会造成 node_modules 半装 + 僵尸进程互相破坏)。

---

# 第二部分:CraftGround 渲染器 — WSL 从零装机

CraftGround(pip 包 `craftground`)是 Minecraft Java 版 + Fabric mod，走 LWJGL/OpenGL 渲染，
与 solaris 的 JS 渲染路完全独立。2026-07-25 在同一 WSL(RTX 3070)实测跑通「每秒截图 + 录像」验收。

## 1. 系统依赖(需 sudo，用户手动执行一次)

CraftGround 官方要求的 apt 包(GL/X dev 头文件 + JDK21 + xvfb):

```bash
!wsl -d Ubuntu -- sudo bash -c "apt-get update && apt-get install -y openjdk-21-jdk python3-pip git libgl1-mesa-dev libegl1-mesa-dev libglew-dev libglu1-mesa-dev xorg-dev libglfw3-dev xvfb"
```

## 2. Python 环境(免 sudo)

系统 python 缺 ensurepip(装 python3-venv 又要 sudo)。用 get-pip.py 给 venv 灌 pip：

```bash
python3 -m venv ~/mc-test/venv           # 若无
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o ~/mc-test/get-pip.py
~/mc-test/venv/bin/python ~/mc-test/get-pip.py -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 3. 装 craftground(脚本 `_install_craftground.sh`)

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
~/mc-test/venv/bin/python -m pip install --upgrade cmake   # 官方要求最新，不能用 apt 的
~/mc-test/venv/bin/python -m pip install craftground        # 连 torch+CUDA13 全套约 2-3GB
~/mc-test/venv/bin/python -m pip install opencv-python-headless  # 验收脚本要 cv2
```

torch 2.13 的 cu130 库与驱动 595.79(支持 CUDA13.2)兼容，`torch.cuda.is_available()=True`(RTX 3070)。

## 4. 首次环境创建 = Gradle 冷编译 + Minecraft asset 下载(唯一硬阻塞)

`env.reset()` 会 `./gradlew runClient`，Fabric loom 先下全 3911 个 Minecraft 1.21 asset 对象。
**任一个下载失败 → downloadAssets 任务失败 → BUILD FAILED → client 根本不启动**
(表现为死循环 `Waiting for server on port 8000` 后进程静默消失，无 crash report)。
海外源 `resources.download.minecraft.net` 经代理慢，常卡在最后 6 个 17-21MB 的 `.ogg` 背景音乐大文件。

补下缺失 asset(脚本 `_fetch_missing_assets.py`，多镜像 bmclapi/官方源 + 长超时)：
asset 齐全后 client 正常启动，`run/logs/latest.log` 出现 `[Player***] Initialization Done`。

无害噪音(别误判为失败)：`Cannot find vcpkg root`、`Missing/Invalid pack metadata`(server resource pack)、
进程退出时 `ModuleNotFoundError: import of time halted`(__del__ 析构期，reset 已成功)。

## 5. 验收(每秒截图 + 录像)

```bash
xvfb-run -a python -m rl_training_environments.craftground.acceptance_sequence \
  --steps 200 --seed 0 --seconds-per-shot 1.0 --video-fps 10 \
  --output-dir runs/craftground-acceptance
```

产出 `sequence.mp4`(200 帧 640×360) + `shots/shot_<秒>_f<帧>.png`(每秒一张) + `summary.json`。
实测出 9 张每秒截图，全部真实森林渲染(树/草地/HUD)。
