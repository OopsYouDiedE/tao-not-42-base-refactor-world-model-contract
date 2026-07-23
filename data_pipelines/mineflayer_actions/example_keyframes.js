// 示例: 在每个关键动作节点截图, 把"节点观测图 + 该节点之后执行的动作"配对成数据集.
// 产出: 每节点一张 PNG + 一个 manifest.json + 一个可直接看的 gallery.md。
//
// 每个节点结构:
//   { index, image, startTick, action: { type, label, durationTicks, ... } }
// 即: 先在动作发生前截取观测帧, 再执行动作并记录其 start/duration, 二者绑定。

const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const { Vec3 } = require('vec3')
const fs = require('fs')
const path = require('path')
const { ActionRecorder } = require('./action_recorder')
const { FrameCapturer } = require('./frame_capturer')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const OUT_DIR = process.env.OUT_DIR || path.resolve(__dirname, 'sample/keyframes')

const bot = mineflayer.createBot({ host: 'localhost', port: 25567, username: 'KeyFrame', version: '1.16.5' })
bot.loadPlugin(pathfinder)

bot.once('spawn', async () => {
  await sleep(1500)
  const mcData = require('minecraft-data')(bot.version)
  const Item = require('prismarine-item')(bot.version)
  bot.pathfinder.setMovements(new Movements(bot, mcData))

  // 走到平坦草地, 保证放置合法且视野开阔
  const feet0 = bot.entity.position.floored()
  for (let radius = 2; radius <= 8; radius++) {
    let found = null
    for (const [dx, dz] of [[radius, 0], [0, radius], [-radius, 0], [0, -radius]]) {
      const g = bot.blockAt(feet0.offset(dx, -1, dz))
      const b = bot.blockAt(feet0.offset(dx, 0, dz))
      const h = bot.blockAt(feet0.offset(dx, 1, dz))
      if (g && /grass|dirt|stone/.test(g.name) && b?.name === 'air' && h?.name === 'air') { found = feet0.offset(dx, 0, dz); break }
    }
    if (found) { try { await Promise.race([bot.pathfinder.goto(new goals.GoalNear(found.x, found.y, found.z, 1)), sleep(12000)]) } catch (e) {} break }
  }
  await sleep(500)

  fs.mkdirSync(OUT_DIR, { recursive: true })
  const recorder = new ActionRecorder(bot)
  const capturer = new FrameCapturer(bot, { width: 640, height: 360 })
  await bot.look(bot.entity.yaw, 0.7, true)   // 俯视, 地面入镜
  await capturer.ready(7000)

  const nodes = []
  // 在每个动作前截观测帧, 执行动作, 记录动作 -> 配对
  async function nodeStep (name, runAction) {
    const before = recorder.records.length
    const image = `node_${String(nodes.length).padStart(2, '0')}_${name}.png`
    // 忠实截取 bot 当前真实视角 (不人为调整朝向, 保证观测与动作时朝向一致)
    fs.writeFileSync(path.join(OUT_DIR, image), await capturer.snapshot({ settleMs: 1200 }))
    const startTick = bot.time.age
    await runAction()
    const newActions = recorder.records.slice(before)
    nodes.push({ index: nodes.length, name, image, startTick, actions: newActions })
    console.log(`[节点${nodes.length - 1}] ${name} -> ${newActions.map((a) => a.label).join(' ') || '(无)'}`)
    await sleep(300)
  }

  // 备料
  await bot.creative.setInventorySlot(36, new Item(mcData.itemsByName.oak_log.id, 8)); await sleep(400)
  await bot.creative.setInventorySlot(37, new Item(mcData.itemsByName.stone.id, 8)); await sleep(400)

  await nodeStep('move_forward', async () => { bot.setControlState('forward', true); await sleep(600); bot.setControlState('forward', false) })
  await nodeStep('jump', async () => { bot.setControlState('jump', true); await sleep(400); bot.setControlState('jump', false) })
  await nodeStep('turn_camera', async () => { await bot.look(bot.entity.yaw + Math.PI / 2, 0.5, true) })
  await nodeStep('craft_planks', async () => { const r = bot.recipesFor(mcData.itemsByName.oak_planks.id, null, 1, null); if (r.length) await bot.craft(r[0], 2, null) })

  // 放置节点: 探测合法落点
  const stone = bot.inventory.items().find((i) => i.name === 'stone')
  if (stone) await bot.equip(stone, 'hand')
  const feet = bot.entity.position.floored()
  let target = null; let reference = null
  for (let radius = 2; radius <= 4 && !target; radius++) {
    for (const [dx, dz] of [[radius, 0], [0, radius], [-radius, 0], [0, -radius]]) {
      const c = feet.offset(dx, 0, dz); const g = bot.blockAt(feet.offset(dx, -1, dz))
      if (bot.blockAt(c)?.name === 'air' && g && g.name !== 'air') { target = c; reference = g; break }
    }
  }
  if (target) {
    await bot.look(bot.entity.yaw, 0.6, true)
    await nodeStep('place_stone', async () => { await bot.placeBlock(reference, new Vec3(0, 1, 0)); await sleep(400) })
    await nodeStep('dig_stone', async () => { const b = bot.blockAt(target); if (b && b.name !== 'air') { try { await Promise.race([bot.dig(b), sleep(6000)]) } catch (e) {} } })
  }

  // 末帧: 动作完成后的最终观测
  const finalImage = `node_${String(nodes.length).padStart(2, '0')}_final.png`
  fs.writeFileSync(path.join(OUT_DIR, finalImage), await capturer.snapshot({ settleMs: 1200 }))

  recorder.flush()
  capturer.dispose()

  // 写 manifest + gallery.md
  const manifest = { source: 'mineflayer', serverVersion: bot.version, tickRate: 20, finalImage, nodeCount: nodes.length, nodes }
  fs.writeFileSync(path.join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2))

  let md = '# 关键节点动作数据集示例\n\n每个节点: **该节点截取的观测帧** + **此后执行的动作**(含起始 tick 与持续 tick)。\n\n'
  for (const n of nodes) {
    md += `## 节点 ${n.index}: ${n.name}\n\n`
    md += `![${n.name}](${n.image})\n\n`
    md += `- 观测截取于 tick ${n.startTick}\n`
    md += '- 此后执行动作:\n'
    for (const a of n.actions) md += `  - \`${a.label}\` (${a.type}) start=${a.startTick}t dur=${a.durationTicks}t / ${a.durationMs}ms\n`
    md += '\n'
  }
  md += `## 末帧观测\n\n![final](${finalImage})\n\n所有动作执行完毕后的最终视角。\n`
  fs.writeFileSync(path.join(OUT_DIR, 'gallery.md'), md)

  console.log(`\n=== 完成: ${nodes.length} 个节点, 图像+manifest+gallery 已写入 ${OUT_DIR} ===`)
  bot.quit(); await sleep(500); process.exit(0)
})
bot.on('error', (e) => console.log('[error]', e.message))
bot.on('kicked', (r) => console.log('[kicked]', JSON.stringify(r).slice(0, 150)))
setTimeout(() => { console.log('[timeout]'); process.exit(1) }, 120000)
