// 生存模式完整挖矿链 (从零开始): 走路探索 -> 砍树采原木 -> 合成木板/木棍/工作台 ->
// 放置工作台 -> 合成木镐 -> 装备 -> 找石头 -> 用镐挖矿。
// 每个关键节点截取观测帧 (FrameCapturer) 并记录其后动作时长 (ActionRecorder),
// 产出 observation(t)->action(t) 配对数据集: PNG + manifest.json + gallery.md。
//
// 复用: gather.js 的 findBlock/digTarget, craft_pickaxe.js 的 craftItem/放工作台。
// 运行需 xvfb (无头 GL 渲染)。服务器须为生存模式, spawn-protection=0。

const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const { plugin: collectBlock } = require('mineflayer-collectblock')
const { Vec3 } = require('vec3')
const fs = require('fs')
const path = require('path')
const { ActionRecorder } = require('./action_recorder')
const { FrameCapturer } = require('./frame_capturer')

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const OUT_DIR = process.env.OUT_DIR || path.resolve(__dirname, 'sample/mining')
const withTimeout = (p, ms, label) => Promise.race([
  Promise.resolve(p).then(() => 'ok'),
  sleep(ms).then(() => { throw new Error(`${label} 超时(${ms}ms)`) })
])

const bot = mineflayer.createBot({ host: 'localhost', port: 25567, username: 'Miner', version: '1.16.5' })
bot.loadPlugin(pathfinder)
bot.loadPlugin(collectBlock)

const invStr = () => bot.inventory.items().map((i) => `${i.name}x${i.count}`).join(', ') || '(空)'
function findBlock (names, maxDistance = 64) {
  const ids = names.map((n) => bot.registry.blocksByName[n]?.id).filter((x) => x != null)
  return bot.findBlock({ matching: ids, maxDistance, count: 1 })
}

let mcData, capturer, recorder
const nodes = []

// 在动作前截观测帧, 执行动作, 记录该节点期间新增的动作 -> 配对
async function nodeStep (name, runAction, note = '') {
  // 停掉上一步可能仍在后台运行的寻路, 避免其移动/转视角动作渗入本节点
  try { bot.pathfinder.setGoal(null) } catch (e) {}
  bot.clearControlStates()
  await sleep(200)
  const before = recorder.records.length
  const image = `node_${String(nodes.length).padStart(2, '0')}_${name}.png`
  fs.writeFileSync(path.join(OUT_DIR, image), await capturer.snapshot({ settleMs: 1200 }))
  const startTick = bot.time.age
  await runAction()
  const actions = recorder.records.slice(before)
  nodes.push({ index: nodes.length, name, note, image, startTick, actions })
  console.log(`[节点${nodes.length - 1}] ${name} -> ${actions.map((a) => a.label).join(' ') || '(无)'} | ${invStr()}`)
  await sleep(300)
}

async function craftItem (name, count, table) {
  const item = mcData.itemsByName[name]
  const recipes = bot.recipesFor(item.id, null, count, table || null)
  if (!recipes.length) { console.log(`[craft] ${name}: 无配方(材料不足或缺工作台)`); return false }
  try { await bot.craft(recipes[0], count, table || null); return true } catch (e) { console.log(`[craft] ${name} 失败:`, e.message); return false }
}

// 寻路到方块附近 (在节点截图前调用, 寻路的移动/转视角微动作不计入节点)
async function approach (block, label) {
  if (!block) return false
  try {
    await withTimeout(bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, 2)), 15000, `${label}寻路`)
    await sleep(300)
    return true
  } catch (e) { console.log(`[${label}] 寻路失败:`, e.message); return false }
}

// 面向并挖掉方块 (只含"看向+挖"两个核心动作; 掉落物靠近后自动拾取, 不额外寻路回捡以免污染节点)
async function faceDig (block, label) {
  const fresh = bot.blockAt(block.position) || block
  try {
    await bot.lookAt(fresh.position.offset(0.5, 0.5, 0.5), true)
    await withTimeout(bot.dig(fresh), 10000, `${label}挖掘`)
  } catch (e) { console.log(`[${label}] 挖掘失败:`, e.message) }
  await sleep(500)   // 停一下让脚下掉落物自动进包
}

bot.once('spawn', async () => {
  await sleep(1500)
  mcData = require('minecraft-data')(bot.version)
  const moves = new Movements(bot, mcData)
  moves.canDig = true
  bot.pathfinder.setMovements(moves)
  console.log('[spawn]', bot.entity.position, bot.game.gameMode)

  // 清空库存与地面散落物, 确保"从零开始"(需 OP; /clear 不产生落地物, 避免走动时被捡回)
  bot.chat('/clear @s')
  await sleep(500)
  bot.chat('/kill @e[type=item]')   // 清掉之前实验残留的地面掉落物
  await sleep(800)
  console.log('[清库存后]', invStr())

  fs.mkdirSync(OUT_DIR, { recursive: true })
  recorder = new ActionRecorder(bot)
  capturer = new FrameCapturer(bot, { width: 640, height: 360 })
  await capturer.ready(6000)

  // 1) 走路探索: 朝最近的树走一段 (若无树则原地前进)
  const logNames = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log']
  const firstLog = findBlock(logNames, 64)
  await nodeStep('walk_to_tree', async () => {
    if (firstLog) {
      try { await withTimeout(bot.pathfinder.goto(new goals.GoalNear(firstLog.position.x, firstLog.position.y, firstLog.position.z, 3)), 15000, '走向树') } catch (e) {}
    } else {
      bot.setControlState('forward', true); await sleep(1500); bot.setControlState('forward', false)
    }
  }, '朝树移动')

  // 2) 砍树: 采集 3~4 块原木 (先寻路到树下, 再在节点内只记"看向+砍")
  for (let i = 0; i < 4; i++) {
    const log = findBlock(logNames, 48)
    if (!log) break
    await approach(log, `砍树#${i + 1}`)   // 寻路不计入节点
    const fresh = findBlock(logNames, 6) || log
    await nodeStep(`chop_log_${i + 1}`, async () => { await faceDig(fresh, `砍树#${i + 1}`) }, '砍原木')
    if (bot.inventory.items().filter((x) => /_log$/.test(x.name)).reduce((s, x) => s + x.count, 0) >= 3) break
  }

  // 3) 合成: 原木->木板, 木板->木棍, 木板->工作台
  await nodeStep('craft_planks', async () => { await craftItem('oak_planks', 3, null) }, '原木→木板')
  await nodeStep('craft_stick', async () => { await craftItem('stick', 1, null) }, '木板→木棍')
  await nodeStep('craft_table', async () => { await craftItem('crafting_table', 1, null) }, '木板→工作台')

  // 4) 放置工作台
  let placed = null
  await nodeStep('place_table', async () => {
    const tableItem = bot.inventory.items().find((i) => i.name === 'crafting_table')
    if (!tableItem) { console.log('[place] 无工作台可放'); return }
    await bot.equip(tableItem, 'hand')
    // 扩大搜索环 (2..3格), 找"目标为空气 + 下方为实体"的落点; 复杂地形下多试几处
    const base = bot.entity.position.floored()
    const ring = []
    for (let r = 1; r <= 3; r++) for (const [dx, dz] of [[r, 0], [-r, 0], [0, r], [0, -r], [r, r], [-r, -r], [r, -r], [-r, r]]) ring.push([dx, dz])
    for (const [dx, dz] of ring) {
      const target = bot.blockAt(base.offset(dx, 0, dz))
      const ground = bot.blockAt(base.offset(dx, -1, dz))
      if (target?.name === 'air' && ground && ground.name !== 'air' && ground.name !== 'water' && ground.name !== 'lava') {
        await bot.lookAt(base.offset(dx, 0, dz).offset(0.5, 0.5, 0.5), true)
        try { await bot.placeBlock(ground, new Vec3(0, 1, 0)); await sleep(500) } catch (e) { console.log('[place] 尝试失败:', e.message) }
        if (bot.blockAt(base.offset(dx, 0, dz))?.name === 'crafting_table') break
      }
    }
    for (let dx = -4; dx <= 4 && !placed; dx++) for (let dz = -4; dz <= 4 && !placed; dz++) for (let dy = -1; dy <= 1 && !placed; dy++) {
      const b = bot.blockAt(bot.entity.position.offset(dx, dy, dz))
      if (b && b.name === 'crafting_table') placed = b
    }
    console.log('[place] 工作台', placed ? `@ ${placed.position}` : '未定位(将徒手2x2降级)')
  }, '放置工作台')

  // 5) 合成木镐 + 装备
  await nodeStep('craft_pickaxe', async () => { await craftItem('wooden_pickaxe', 1, placed) }, '合成木镐(靠工作台)')
  const pick = bot.inventory.items().find((i) => i.name === 'wooden_pickaxe')
  if (pick) { try { await bot.equip(pick, 'hand') } catch (e) {} }
  console.log(pick ? '=== ✓ 已获得并装备木镐 ===' : '=== ✗ 未获得木镐 ===')

  // 6) 找石头: 若地表无裸露石头, 先下挖穿土层
  const stoneNames = ['stone', 'coal_ore', 'iron_ore', 'andesite', 'diorite', 'granite']
  let stone = findBlock(stoneNames, 24)
  if (!stone) {
    // 阶梯下挖穿土到石头 (复用 mine.js 思路)
    await nodeStep('dig_down_to_stone', async () => {
      for (let d = 0; d < 6; d++) {
        const below = bot.blockAt(bot.entity.position.offset(0, -1, 0))
        if (!below || below.name === 'air' || below.name === 'bedrock') break
        try { await withTimeout(bot.dig(below), 8000, `下挖${below.name}`) } catch (e) { break }
        try { await withTimeout(bot.pathfinder.goto(new goals.GoalBlock(bot.entity.position.floored().x, below.position.y, bot.entity.position.floored().z)), 5000, '下移') } catch (e) {}
        if (/stone|ore|andesite|diorite|granite/.test(below.name)) break
      }
    }, '下挖穿土到石头')
    stone = findBlock(stoneNames, 12)
  }

  // 7) 用镐挖石头: 逐块挖, 每块一个节点 (寻路不计入; 节点内只记"看向+挖", 有真实挖掘时长)
  for (let i = 0; i < 4; i++) {
    const target = findBlock(stoneNames, 16)
    if (!target) { console.log('[挖矿] 附近无更多石头'); break }
    await approach(target, `挖矿#${i + 1}`)   // 寻路不计入节点
    // 挖前确保木镐在手 (寻路/合成后手持可能被换下, 否则会退化成徒手挖: 慢且石头无掉落)
    const heldPick = bot.inventory.items().find((it) => it.name === 'wooden_pickaxe')
    if (heldPick && bot.heldItem?.name !== 'wooden_pickaxe') { try { await bot.equip(heldPick, 'hand') } catch (e) {} }
    console.log(`[挖矿#${i + 1}] 手持: ${bot.heldItem?.name || '空手'}`)
    const fresh = findBlock(stoneNames, 6) || target
    await nodeStep(`mine_stone_${i + 1}`, async () => { await faceDig(fresh, `挖矿#${i + 1}`) }, '用木镐挖石头')
  }

  // 末帧
  const finalImage = `node_${String(nodes.length).padStart(2, '0')}_final.png`
  fs.writeFileSync(path.join(OUT_DIR, finalImage), await capturer.snapshot({ settleMs: 1200 }))
  recorder.flush(); capturer.dispose()

  // 写 manifest + gallery
  const manifest = { source: 'mineflayer', serverVersion: bot.version, gameMode: 'survival', tickRate: 20, task: 'survival_mining_chain', finalImage, nodeCount: nodes.length, finalInventory: invStr(), nodes }
  fs.writeFileSync(path.join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2))

  let md = '# 生存模式挖矿链 · 关键节点数据集\n\n从零开始的完整链路: 走路 → 砍树 → 合成(木板/木棍/工作台) → 放工作台 → 合成木镐 → 用镐挖石头。\n每个节点: **该节点观测帧** + **此后执行的动作**(含起始 tick 与持续 tick)。\n\n'
  for (const n of nodes) {
    md += `## 节点 ${n.index}: ${n.name}${n.note ? ` (${n.note})` : ''}\n\n![${n.name}](${n.image})\n\n- 观测截取于 tick ${n.startTick}\n- 此后执行动作:\n`
    if (n.actions.length === 0) md += '  - (无记录动作)\n'
    for (const a of n.actions) md += `  - \`${a.label}\` (${a.type}) start=${a.startTick}t dur=${a.durationTicks}t / ${a.durationMs}ms${a.detail?.block ? ` [${a.detail.block}]` : ''}\n`
    md += '\n'
  }
  md += `## 末帧\n\n![final](${finalImage})\n\n最终库存: ${invStr()}\n`
  fs.writeFileSync(path.join(OUT_DIR, 'gallery.md'), md)

  console.log(`\n=== 完成: ${nodes.length} 节点, 最终库存: ${invStr()} ===`)
  bot.quit(); await sleep(500); process.exit(0)
})
bot.on('error', (e) => console.log('[error]', e.message))
bot.on('kicked', (r) => console.log('[kicked]', JSON.stringify(r).slice(0, 150)))
setTimeout(() => { console.log('[timeout] 总超时'); process.exit(1) }, 280000)
