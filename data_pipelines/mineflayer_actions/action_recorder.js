// 动作记录器: 为 mineflayer bot 的每个动作记录 startTick 与 durationTicks。
// 时间基准用 bot.time.age (Minecraft 世界年龄, 20 tick/秒)。
// 覆盖 6 类: 移动(move) / 转视角(look) / 跳跃(jump) / 合成(craft) / 放置(place) / 破坏(dig)。
//
// 输出动作串对齐 net/action_token_codec.py 的规范动作:
//   移动 F/B/L/R, jump/sneak/sprint, 破坏 attack, 使用/交互 use, 相机 cam(dYaw,dPitch)。
// 每条记录形如:
//   { type, label, startTick, endTick, durationTicks, startMs, durationMs, detail }

class ActionRecorder {
  constructor (bot) {
    this.bot = bot
    this.records = []
    this._open = new Map()   // 持续型动作的进行中状态: key -> {startTick, startMs, detail}
    this._install()
  }

  tick () { return this.bot.time?.age ?? 0 }
  now () { return Date.now() }

  _emit (type, label, startTick, startMs, detail = {}) {
    const endTick = this.tick()
    const rec = {
      type, label,
      startTick, endTick,
      durationTicks: Math.max(0, endTick - startTick),
      startMs, durationMs: this.now() - startMs,
      detail
    }
    this.records.push(rec)
    return rec
  }

  // 持续型动作: begin/end 配对 (移动键、跳跃、潜行、冲刺)
  begin (key, detail = {}) {
    if (this._open.has(key)) return
    this._open.set(key, { startTick: this.tick(), startMs: this.now(), detail })
  }

  end (key) {
    const s = this._open.get(key)
    if (!s) return null
    this._open.delete(key)
    return this._emit('sustained', key, s.startTick, s.startMs, s.detail)
  }

  // 瞬时/带时长动作: 用 wrap 包裹一个 promise, 自动记录起止与成败
  async wrap (type, label, fn, detail = {}) {
    const startTick = this.tick(); const startMs = this.now()
    let ok = true; let err = null
    try { await fn() } catch (e) { ok = false; err = e.message }
    return this._emit(type, label, startTick, startMs, { ...detail, ok, err })
  }

  _install () {
    const bot = this.bot

    // 1) 移动 + 姿态: 拦截 setControlState, 起停即记录持续时长
    const controlLabel = { forward: 'F', back: 'B', left: 'L', right: 'R', jump: 'jump', sneak: 'sneak', sprint: 'sprint' }
    const originalSetControlState = bot.setControlState.bind(bot)
    bot.setControlState = (control, state) => {
      const label = controlLabel[control] || control
      if (state) this.begin(label, { control })
      else this.end(label)
      return originalSetControlState(control, state)
    }

    // 2) 转视角: 拦截 look, 记录相机偏移 cam(dYaw,dPitch) (角度近似, 单位度)
    const originalLook = bot.look.bind(bot)
    bot.look = async (yaw, pitch, force) => {
      const startTick = this.tick(); const startMs = this.now()
      const yawBefore = bot.entity.yaw; const pitchBefore = bot.entity.pitch
      await originalLook(yaw, pitch, force)
      const deltaYaw = Math.round((yaw - yawBefore) * 180 / Math.PI)
      const deltaPitch = Math.round((pitch - pitchBefore) * 180 / Math.PI)
      const sign = (v) => (v >= 0 ? '+' : '') + v
      this._emit('look', `cam(${sign(deltaYaw)},${sign(deltaPitch)})`, startTick, startMs, { deltaYaw, deltaPitch })
    }

    // 3) 破坏: dig
    const originalDig = bot.dig.bind(bot)
    bot.dig = async (block, ...rest) => {
      const startTick = this.tick(); const startMs = this.now()
      const blockName = block?.name
      let ok = true; let err = null
      try { await originalDig(block, ...rest) } catch (e) { ok = false; err = e.message }
      this._emit('dig', 'attack', startTick, startMs, { block: blockName, ok, err })
    }

    // 4) 放置: placeBlock。blockUpdate 确认事件在部分服务器版本常超时,
    //    这里记录耗时并保留 err, 方块是否真放上由调用方用 blockAt 校验。
    const originalPlaceBlock = bot.placeBlock.bind(bot)
    bot.placeBlock = async (referenceBlock, faceVector, ...rest) => {
      const startTick = this.tick(); const startMs = this.now()
      let ok = true; let err = null
      try { await originalPlaceBlock(referenceBlock, faceVector, ...rest) } catch (e) { ok = false; err = e.message }
      this._emit('place', 'use', startTick, startMs, { reference: referenceBlock?.name, ok, err })
    }

    // 5) 合成: craft
    const originalCraft = bot.craft.bind(bot)
    bot.craft = async (recipe, count, craftingTable, ...rest) => {
      const startTick = this.tick(); const startMs = this.now()
      const resultId = recipe?.result?.id
      const resultName = resultId != null ? (bot.registry?.items?.[resultId]?.name ?? resultId) : 'unknown'
      let ok = true; let err = null
      try { await originalCraft(recipe, count, craftingTable, ...rest) } catch (e) { ok = false; err = e.message }
      this._emit('craft', `craft:${resultName}`, startTick, startMs, { result: resultName, count, ok, err })
    }
  }

  // 收尾: 关闭所有仍打开的持续动作
  flush () { for (const key of [...this._open.keys()]) this.end(key) }

  toJSON (meta = {}) {
    return { ...meta, tickRate: 20, actionCount: this.records.length, records: this.records }
  }
}

module.exports = { ActionRecorder }
