// 采集入口: 连接 Java Minecraft 服务器, 驱动 mineflayer bot 执行一段脚本化动作,
// 用 ActionRecorder 记录每个动作的 startTick / durationTicks, 落盘为 JSON。
//
// 用法:
//   node record_session.js --host localhost --port 25565 --version 1.16.5 \
//                          --output sample/session_actions.json
//
// 前置: 目标服务器已启动, 创造模式, spawn-protection=0 (否则放置/破坏在出生点被拒)。
// 详见同目录 AGENTS.md。

const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const { Vec3 } = require('vec3')
const fs = require('fs')
const path = require('path')
const { ActionRecorder } = require('./action_recorder')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function parseArguments (argv) {
  const options = { host: 'localhost', port: 25565, version: '1.16.5', username: 'Recorder', output: 'session_actions.json' }
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '')
    const value = argv[i + 1]
    if (key === 'port') options.port = parseInt(value, 10)
    else if (key in options) options[key] = value
  }
  return options
}

async function main () {
  const options = parseArguments(process.argv)
  const bot = mineflayer.createBot({ host: options.host, port: options.port, username: options.username, version: options.version })
  bot.loadPlugin(pathfinder)

  bot.once('spawn', async () => {
    await sleep(1500)
    const mcData = require('minecraft-data')(bot.version)
    const Item = require('prismarine-item')(bot.version)
    bot.pathfinder.setMovements(new Movements(bot, mcData))

    // 走到一块平坦地面, 保证放置有合法落点 (避开先前实验破坏的地形)
    const feet0 = bot.entity.position.floored()
    let flat = null
    for (let radius = 2; radius <= 8 && !flat; radius++) {
      for (const [dx, dz] of [[radius, 0], [0, radius], [-radius, 0], [0, -radius]]) {
        const ground = bot.blockAt(feet0.offset(dx, -1, dz))
        const body = bot.blockAt(feet0.offset(dx, 0, dz))
        const head = bot.blockAt(feet0.offset(dx, 1, dz))
        if (ground && /grass|dirt|stone/.test(ground.name) && body?.name === 'air' && head?.name === 'air') {
          flat = feet0.offset(dx, 0, dz); break
        }
      }
    }
    if (flat) {
      try { await Promise.race([bot.pathfinder.goto(new goals.GoalNear(flat.x, flat.y, flat.z, 1)), sleep(15000)]) } catch (e) {}
    }
    await sleep(500)

    const recorder = new ActionRecorder(bot)
    const startTick = bot.time.age

    // 1) 移动: 前进约 1.5s
    bot.setControlState('forward', true); await sleep(1500); bot.setControlState('forward', false)
    // 2) 跳跃: 约 0.4s
    bot.setControlState('jump', true); await sleep(400); bot.setControlState('jump', false)
    // 3) 转视角
    await bot.look(bot.entity.yaw + Math.PI / 2, 0.3, true); await sleep(300)

    // 备料 (合成与放置)
    await bot.creative.setInventorySlot(36, new Item(mcData.itemsByName.oak_log.id, 8)); await sleep(400)
    await bot.creative.setInventorySlot(37, new Item(mcData.itemsByName.stone.id, 8)); await sleep(400)

    // 4) 合成: 原木 -> 木板
    const planksRecipes = bot.recipesFor(mcData.itemsByName.oak_planks.id, null, 1, null)
    if (planksRecipes.length) await bot.craft(planksRecipes[0], 2, null)
    await sleep(500)

    // 5) 放置: 在 2~4 格外的"空气格 + 下方实体"处放石头
    const stoneItem = bot.inventory.items().find((i) => i.name === 'stone')
    if (stoneItem) await bot.equip(stoneItem, 'hand')
    const feet = bot.entity.position.floored()
    let target = null; let reference = null
    for (let radius = 2; radius <= 4 && !target; radius++) {
      for (const [dx, dz] of [[radius, 0], [0, radius], [-radius, 0], [0, -radius], [radius, radius], [-radius, -radius]]) {
        const candidate = feet.offset(dx, 0, dz)
        const ground = bot.blockAt(feet.offset(dx, -1, dz))
        if (bot.blockAt(candidate)?.name === 'air' && ground && ground.name !== 'air') { target = candidate; reference = ground; break }
      }
    }
    if (target) {
      await bot.look(bot.entity.yaw, 0.6, true)
      await bot.placeBlock(reference, new Vec3(0, 1, 0))
      await sleep(400)
    }

    // 6) 破坏: 挖掉刚放的石头, 否则挖脚下方块
    let toDig = target ? bot.blockAt(target) : null
    if (!toDig || toDig.name === 'air') toDig = bot.blockAt(feet.offset(0, -1, 0))
    if (toDig && toDig.name !== 'air') {
      try { await Promise.race([bot.dig(toDig), sleep(6000)]) } catch (e) {}
    }

    recorder.flush()

    const payload = recorder.toJSON({
      source: 'mineflayer',
      serverVersion: bot.version,
      task: 'six_action_smoke',
      sessionStartTick: startTick
    })
    const outputPath = path.isAbsolute(options.output) ? options.output : path.resolve(__dirname, options.output)
    fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2))
    console.log(`已记录 ${payload.actionCount} 条动作 -> ${outputPath}`)
    for (const action of payload.records) {
      console.log(`  [${action.type}] ${action.label}  start=${action.startTick}t dur=${action.durationTicks}t (${action.durationMs}ms)`)
    }

    bot.quit()
    await sleep(500)
    process.exit(0)
  })

  bot.on('error', (e) => console.log('[error]', e.message))
  bot.on('kicked', (reason) => console.log('[kicked]', JSON.stringify(reason).slice(0, 200)))
  setTimeout(() => { console.log('[timeout] 会话超时退出'); process.exit(1) }, 90000)
}

main()
