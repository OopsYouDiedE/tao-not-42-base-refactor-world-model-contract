// 单帧截图器: 基于 prismarine-viewer 的无头渲染, 按需对 bot 当前第一人称视角截 PNG.
// 与连续录制 mp4 不同, 这里每个动作节点主动截一帧观测 (obs), 与其后的动作配对.
//
// 用法:
//   const cap = new FrameCapturer(bot, { width: 640, height: 360 })
//   await cap.ready()                 // 等世界 mesh 生成
//   const pngBuffer = cap.snapshot()  // 截取当前视角
//   cap.dispose()
//
// 需在有虚拟显示的环境运行 (xvfb), 依赖 node-canvas-webgl。

global.THREE = require('three')
global.Worker = require('worker_threads').Worker
const { createCanvas } = require('node-canvas-webgl/lib')
const { WorldView, Viewer } = require('prismarine-viewer/viewer')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

class FrameCapturer {
  constructor (bot, { width = 640, height = 360, viewDistance = 6 } = {}) {
    this.bot = bot
    this.canvas = createCanvas(width, height)
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas })
    this.viewer = new Viewer(this.renderer)
    if (!this.viewer.setVersion(bot.version)) throw new Error(`viewer 不支持版本 ${bot.version}`)
    this.worldView = new WorldView(bot.world, viewDistance, bot.entity.position)
    this.viewer.listen(this.worldView)
  }

  // 持续驱动渲染循环, 让 worker 线程把网格化结果上传到场景 (单纯 sleep 不推进 mesh)。
  async _pump (durationMs, stepMs = 100) {
    const steps = Math.max(1, Math.round(durationMs / stepMs))
    for (let i = 0; i < steps; i++) {
      this.viewer.update()
      this.renderer.render(this.viewer.scene, this.viewer.camera)
      await sleep(stepMs)
    }
  }

  // 初始化世界视图并驱动 mesh 生成 (worker 线程异步网格化, 需持续渲染推进)
  async ready (meshWaitMs = 6000) {
    await this.worldView.init(this.bot.entity.position)
    await this._pump(meshWaitMs)
  }

  // 截取 bot 当前第一人称视角, 返回 PNG Buffer。
  // settleMs > 0 时先更新位置并持续渲染, 让周边 chunk 网格铺满, 避免拍到碎片/空白。
  async snapshot ({ settleMs = 0 } = {}) {
    const { position, yaw, pitch } = this.bot.entity
    this.worldView.updatePosition(position)
    this.viewer.setFirstPersonCamera(position, yaw, pitch)
    if (settleMs > 0) await this._pump(settleMs)
    this.viewer.update()
    this.renderer.render(this.viewer.scene, this.viewer.camera)
    return this.canvas.toBuffer('image/png')
  }

  dispose () {
    try { this.renderer.dispose() } catch (e) {}
  }
}

module.exports = { FrameCapturer }
