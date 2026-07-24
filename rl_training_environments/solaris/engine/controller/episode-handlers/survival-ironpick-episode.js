// survival-ironpick-episode.js
//
// Single-bot, from-scratch survival chain recorded through the solaris headless
// renderer: gather wood -> craft planks/sticks/crafting table -> wooden pickaxe
// -> mine stone -> stone pickaxe -> mine coal + iron ore -> smelt iron -> craft
// iron pickaxe. Every step uses REAL mineflayer actions (dig / craft / furnace /
// pathfinder). No RCON `give` of items or `tp` — run with --no_cheat_items 1 so
// the per-episode inventory grant is skipped and the bot truly starts empty.
//
// Designed for single-bot runs (LoopbackCoordinator). Progress-driven: each step
// completes when the target item count is reached, with per-step timeouts; any
// hard failure is logged and the chain stops gracefully (still produces a clip).

const { Vec3 } = require("vec3");
const {
  Movements,
  goals: { GoalNear, GoalGetToBlock, GoalBlock },
} = require("mineflayer-pathfinder");

const { digWithTimeout } = require("../primitives/digging");
const { ensureItemInHand, unequipHand } = require("../primitives/items");
const {
  gotoWithTimeout,
  getScaffoldingBlockIds,
  stopAll,
  sleep,
  lookAtSmooth,
} = require("../primitives/movement");
const { placeAt } = require("../primitives/building");
const { BaseEpisode } = require("./base-episode");

// ---- tunables ----
const SEARCH_DIST = 64; // findBlock max distance
const REACH_TIMEOUT_MS = 20000; // pathfind-to-block timeout
const DIG_TIMEOUT_MS = 15000; // single-block dig timeout
const PICKUP_TICKS = 12; // ticks to wait on top of a drop
const STEP_MAX_TRIES = 40; // safety cap on collect-loop iterations

/**
 * Build a pathfinder Movements tuned for gathering: allow digging so the bot can
 * tunnel toward targets, but keep it non-destructive-happy (normal costs).
 */
function makeGatherMovements(bot, mcData) {
  const m = new Movements(bot, mcData);
  m.canDig = true;
  m.allowParkour = true;
  m.allowSprinting = true;
  m.scafoldingBlocks = getScaffoldingBlockIds(mcData);
  m.maxDropDown = 16;
  m.dontCreateFlow = true;
  m.dontMineUnderFallingBlock = true;
  return m;
}

function invCount(bot, mcData, itemName) {
  const it = mcData.itemsByName[itemName];
  if (!it) return 0;
  return bot.inventory.count(it.id, null);
}

/** Sum inventory counts across several item names (e.g. all plank variants). */
function invCountAny(bot, mcData, names) {
  return names.reduce((s, n) => s + invCount(bot, mcData, n), 0);
}

/**
 * Collect `target` blocks matching any of `blockNames`.
 * findBlock -> GoalGetToBlock (stop adjacent, in reach) -> auto tool -> dig ->
 * walk onto the drop (GoalNear r=1) -> verify via inventory count.
 * Returns the number collected (best-effort; may be < target on timeout).
 */
async function collect(bot, mcData, blockNames, target, opts = {}) {
  const maxDist = opts.maxDist || SEARCH_DIST;
  const ids = blockNames
    .map((n) => mcData.blocksByName[n]?.id)
    .filter((x) => x != null);
  if (ids.length === 0) {
    console.log(`[${bot.username}] collect: no such blocks ${blockNames}`);
    return 0;
  }
  // item names produced (for verifying pickup): usually same-named item, but
  // stone->cobblestone, *_ore->raw_* differ, so caller passes verifyItems.
  const verifyItems = opts.verifyItems || blockNames;
  const startCount = invCountAny(bot, mcData, verifyItems);
  let tries = 0;
  // Blacklist positions we tried but couldn't reach/dig, so findBlock's
  // "nearest first" doesn't hand us the same unreachable block forever.
  const blacklist = new Set();
  const key = (v) => `${v.x},${v.y},${v.z}`;

  while (invCountAny(bot, mcData, verifyItems) - startCount < target) {
    if (tries++ >= STEP_MAX_TRIES) {
      console.log(`[${bot.username}] collect(${blockNames}): hit try cap`);
      break;
    }
    // find nearest matching block that isn't blacklisted
    const candidates = bot.findBlocks({
      matching: ids,
      maxDistance: maxDist,
      count: 24,
    });
    const p = candidates.find((c) => !blacklist.has(key(c)));
    if (!p) {
      console.log(
        `[${bot.username}] collect(${blockNames}): no reachable candidates within ${maxDist}`,
      );
      break;
    }
    try {
      await gotoWithTimeout(
        bot,
        new GoalGetToBlock(p.x, p.y, p.z),
        { timeoutMs: REACH_TIMEOUT_MS, stopOnTimeout: true },
      );
    } catch (e) {
      console.log(`[${bot.username}] reach failed: ${e.message}`);
    }
    // re-fetch the block (world may have changed) and dig
    const target2 = bot.blockAt(p);
    if (!target2 || !ids.includes(target2.type)) {
      continue; // already gone / mismatched; find next
    }
    // Verify the bot can actually reach the block before digging; if not,
    // blacklist this position and look for another (avoids a 15s dig timeout
    // on an unreachable tree-top / walled-off block).
    if (!bot.canDigBlock(target2)) {
      const d = bot.entity.position.distanceTo(p.offset(0.5, 0.5, 0.5));
      console.log(
        `[${bot.username}] out of reach (dist ${d.toFixed(1)}), skipping ${target2.name} @ ${p}`,
      );
      blacklist.add(key(p));
      continue;
    }
    // digWithTimeout resolves undefined on success and THROWS on timeout/error,
    // so wrap it: a failed/timed-out dig should skip this block, not kill the chain.
    try {
      await digWithTimeout(bot, target2, { timeoutMs: DIG_TIMEOUT_MS });
    } catch (e) {
      console.log(`[${bot.username}] dig failed at ${p}: ${e.message}`);
      blacklist.add(key(p));
      continue;
    }
    blacklist.add(key(p)); // mined; don't revisit this exact spot
    // walk onto the drop so it enters the inventory (no auto-pickup in mineflayer)
    try {
      await gotoWithTimeout(bot, new GoalNear(p.x, p.y, p.z, 1), {
        timeoutMs: 6000,
        stopOnTimeout: true,
      });
    } catch (e) {
      /* fine */
    }
    await bot.waitForTicks(PICKUP_TICKS);
    const have = invCountAny(bot, mcData, verifyItems) - startCount;
    console.log(
      `[${bot.username}] collect(${blockNames}) progress: ${have}/${target}`,
    );
  }
  return invCountAny(bot, mcData, verifyItems) - startCount;
}

/**
 * Return a crafting table Block the bot can currently reach: a nearby existing
 * one if found & reachable, otherwise place a fresh one where the bot stands
 * (the bot keeps planks, and crafts a new table if it has none). This makes the
 * chain robust to having tunneled far from the original surface table.
 */
async function ensureTableNearby(bot, mcData) {
  const tableId = mcData.blocksByName.crafting_table?.id;
  const near = tableId != null ? bot.findBlock({ matching: tableId, maxDistance: 4 }) : null;
  if (near && bot.canDigBlock(near)) return near;
  // need a new one: craft it if absent (2 planks -> 1 table via a plank; recipe needs 4 planks)
  if (invCount(bot, mcData, "crafting_table") < 1) {
    const plank = ["oak_planks","birch_planks","spruce_planks","jungle_planks","acacia_planks","dark_oak_planks"].find((p) => invCount(bot, mcData, p) >= 4);
    if (plank) {
      try { await craft(bot, mcData, "crafting_table", 1, null); } catch (e) { /* fall through */ }
    }
  }
  if (invCount(bot, mcData, "crafting_table") >= 1) {
    return await placeAndFind(bot, mcData, "crafting_table");
  }
  throw new Error("no crafting table reachable and cannot craft one");
}

/** Craft `count` of itemName, optionally at a crafting table Block. */
async function craft(bot, mcData, itemName, count, tableBlock) {
  const item = mcData.itemsByName[itemName];
  if (!item) throw new Error(`no such item ${itemName}`);
  const recipes = bot.recipesFor(item.id, null, 1, tableBlock || null);
  if (!recipes || recipes.length === 0) {
    throw new Error(
      `no craftable recipe for ${itemName} (missing ingredients or table?)`,
    );
  }
  await bot.craft(recipes[0], count, tableBlock || null);
  console.log(
    `[${bot.username}] crafted ${count}x ${itemName} (now ${invCount(bot, mcData, itemName)})`,
  );
}

/**
 * Dig a staircase downward, collecting whatever we pass through, until we've
 * accumulated `target` of the wanted items OR reached maxDepth. Real players do
 * this to reach stone/ores buried under a grass-and-dirt surface world, where
 * findBlock+pathfind-to-a-buried-block just times out (the bot can't reach a
 * block encased in terrain). We mine the block below-and-forward each step so we
 * descend a walkable staircase rather than pillar-dropping into a hole.
 */
async function digDownForBlocks(bot, mcData, verifyItems, target, maxDepth = 30) {
  const startCount = invCountAny(bot, mcData, verifyItems);
  const have = () => invCountAny(bot, mcData, verifyItems) - startCount;
  const dirs = [new Vec3(1, 0, 0), new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(0, 0, -1)];
  let steps = 0;
  while (have() < target && steps < maxDepth) {
    steps++;
    const feet = bot.entity.position.floored();
    const dir = dirs[steps % 4]; // rotate heading so we spiral down, staying reachable
    // dig the block diagonally below-forward, then the one below it, forming stairs
    const forwardDown = feet.plus(dir).offset(0, -1, 0);
    const straightDown = feet.offset(0, -1, 0);
    for (const cell of [forwardDown, straightDown]) {
      const blk = bot.blockAt(cell);
      if (!blk || isAiry(blk)) continue;
      if (blk.name.includes("lava") || blk.name.includes("water")) {
        console.log(`[${bot.username}] digDown: hit ${blk.name}, stopping`);
        return have();
      }
      try {
        await bot.tool.equipForBlock(blk, {});
      } catch (e) { /* best effort */ }
      try {
        await bot.lookAt(cell.offset(0.5, 0.5, 0.5), true);
        await digWithTimeout(bot, blk, { timeoutMs: DIG_TIMEOUT_MS });
      } catch (e) {
        console.log(`[${bot.username}] digDown dig failed: ${e.message}`);
      }
    }
    // step down into the freshly dug space
    try {
      await gotoWithTimeout(bot, new GoalNear(feet.plus(dir).x, feet.y - 1, feet.plus(dir).z, 1), { timeoutMs: 5000, stopOnTimeout: true });
    } catch (e) { /* gravity / partial move is fine */ }
    await bot.waitForTicks(6);
    console.log(`[${bot.username}] digDown progress ${have()}/${target} (depth step ${steps}, y=${Math.floor(bot.entity.position.y)})`);
  }
  return have();
}

const AIRY = new Set(["air", "cave_air", "void_air", "short_grass", "tall_grass", "grass", "fern", "snow"]);
function isAiry(block) {
  return !block || AIRY.has(block.name);
}
function isSolidGround(block) {
  return block && !isAiry(block) && block.boundingBox === "block";
}

/**
 * Place `itemName` on a valid ground spot near the bot and return the placed
 * Block. Strategy: scan the ground ring around the bot for an AIR cell whose
 * block-below is SOLID (a real reference face for placeBlock); clear leaves/grass
 * if they block the target cell. Falls back to scanning for the placed block.
 * Uses primitives/placeAt only (no RCON).
 */
async function placeAndFind(bot, mcData, itemName) {
  await ensureItemInHand(bot, itemName);
  const feet = bot.entity.position.floored();
  // candidate XZ offsets around the bot (ring of radius 1..2)
  const ring = [
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
    [2, 0], [-2, 0], [0, 2], [0, -2],
  ];
  for (const [dx, dz] of ring) {
    // search a few Y levels around feet for "air cell with solid block below"
    for (const dy of [0, -1, 1]) {
      const cell = feet.offset(dx, dy, dz);
      const below = cell.offset(0, -1, 0);
      const cellBlk = bot.blockAt(cell);
      const belowBlk = bot.blockAt(below);
      if (!isSolidGround(belowBlk)) continue;
      // clear soft obstruction (grass/leaves) occupying the target cell
      if (cellBlk && !isAiry(cellBlk)) {
        if (cellBlk.name.includes("leaves") || cellBlk.name.includes("grass")) {
          try { await digWithTimeout(bot, cellBlk, { timeoutMs: 6000 }); } catch (e) { continue; }
        } else {
          continue; // solid non-clearable block occupies the cell
        }
      }
      const placed = await placeAt(bot, cell, itemName, { tries: 3 });
      if (placed) {
        await sleep(300);
        const blk = bot.blockAt(cell);
        if (blk && blk.name === itemName) {
          console.log(`[${bot.username}] placed ${itemName} at ${cell}`);
          return blk;
        }
      }
    }
  }
  // last resort: maybe it landed somewhere slightly off — scan for it
  const id = mcData.blocksByName[itemName]?.id;
  if (id != null) {
    const found = bot.findBlock({ matching: id, maxDistance: 6 });
    if (found) {
      console.log(`[${bot.username}] found placed ${itemName} at ${found.position}`);
      return found;
    }
  }
  throw new Error(`failed to place ${itemName}`);
}

/** Smelt raw_iron -> iron_ingot in a placed furnace. fuel = any fuel item name. */
async function smeltIron(bot, mcData, furnaceBlock, fuelName, want) {
  const furnace = await bot.openFurnace(furnaceBlock);
  const fuel = mcData.itemsByName[fuelName];
  const rawIron = mcData.itemsByName.raw_iron;
  try {
    await furnace.putFuel(fuel.id, null, Math.max(1, Math.ceil(want / 8)));
    await furnace.putInput(rawIron.id, null, want);
    console.log(`[${bot.username}] furnace loaded: ${want} raw_iron + ${fuelName}`);
    // wait for output, polling the update event with a hard timeout
    await new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        const out = furnace.outputItem();
        if (out && out.count >= want) {
          done = true;
          resolve();
        }
      };
      furnace.on("update", finish);
      const timer = setInterval(finish, 2000);
      setTimeout(() => {
        done = true;
        clearInterval(timer);
        resolve();
      }, 60000);
    });
    const collected = [];
    let out;
    while ((out = furnace.outputItem())) {
      collected.push(await furnace.takeOutput());
      if (!furnace.outputItem()) break;
    }
    console.log(
      `[${bot.username}] smelted ${collected.reduce((s, i) => s + (i?.count || 0), 0)} iron ingot(s)`,
    );
  } finally {
    furnace.close();
  }
}

/**
 * The survival chain. Each step is best-effort; on hard failure we log and stop
 * (still yields a partial clip). Returns a summary object.
 */
async function runSurvivalChain(bot, args) {
  const mcData = require("minecraft-data")(bot.version);
  bot.pathfinder.setMovements(makeGatherMovements(bot, mcData));

  const summary = { reached: "start", ironIngots: 0, ironPickaxe: false };
  const log = (m) => console.log(`[${bot.username}] 🎯 ${m}`);

  const PLANKS = ["oak_planks", "birch_planks", "spruce_planks", "jungle_planks", "acacia_planks", "dark_oak_planks"];
  const LOGS = ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log"];
  // planks item name used for crafting derivatives depends on the log type; we
  // craft whichever plank we actually have. Helper to pick the first we own.
  const firstOwnedPlank = () =>
    PLANKS.find((p) => invCount(bot, mcData, p) > 0);

  try {
    // 1) WOOD: gather 6 logs -> ~24 planks. We need planks for: crafting table(4),
    // wooden pickaxe(3), sticks (for 3 pickaxes = 6 sticks = 3 planks), a REBUILT
    // table after tunneling underground(4), furnace uses cobble not wood. 6 logs
    // gives comfortable margin so the chain never stalls for lack of a table.
    log("STEP 1: gathering wood");
    const gotWood = await collect(bot, mcData, LOGS, 6, { verifyItems: LOGS });
    if (gotWood < 1) throw new Error("no wood gathered");
    summary.reached = "wood";

    // 2) PLANKS + STICKS + TABLE
    log("STEP 2: crafting planks, sticks, table");
    // craft planks from each log we hold (recipe is per-species; craft the ones we own)
    for (const logName of LOGS) {
      if (invCount(bot, mcData, logName) > 0) {
        const plankName = logName.replace("_log", "_planks");
        const times = invCount(bot, mcData, logName);
        try { await craft(bot, mcData, plankName, times, null); } catch (e) { /* skip */ }
      }
    }
    const plankName = firstOwnedPlank();
    if (!plankName) throw new Error("no planks after crafting");
    const totalPlanks = () => PLANKS.reduce((s, p) => s + invCount(bot, mcData, p), 0);
    log(`have ${totalPlanks()} planks after conversion`);
    // sticks: craft twice (2 planks -> 4 sticks each) to stockpile ~6+ for 3 pickaxes
    for (let i = 0; i < 2; i++) {
      try { await craft(bot, mcData, "stick", 1, null); } catch (e) { log("stick craft: " + e.message); }
    }
    await craft(bot, mcData, "crafting_table", 1, null);
    summary.reached = "planks";

    // place table and get within reach
    const table = await placeAndFind(bot, mcData, "crafting_table");
    await gotoWithTimeout(bot, new GoalGetToBlock(table.position.x, table.position.y, table.position.z), { timeoutMs: REACH_TIMEOUT_MS });

    // 3) WOODEN PICKAXE
    log("STEP 3: crafting wooden pickaxe");
    await craft(bot, mcData, "wooden_pickaxe", 1, table);
    await ensureItemInHand(bot, "wooden_pickaxe");
    summary.reached = "wooden_pickaxe";

    // 4) STONE — try surface first, else dig a staircase down to the stone layer.
    log("STEP 4: mining stone");
    let gotStone = await collect(bot, mcData, ["stone"], 3, { verifyItems: ["cobblestone"], maxDist: 12 });
    if (gotStone < 3) {
      log(`only ${gotStone} cobble from surface; digging down for stone`);
      gotStone += await digDownForBlocks(bot, mcData, ["cobblestone"], 3 - gotStone, 25);
    }
    if (gotStone < 3) throw new Error(`only ${gotStone} cobblestone`);
    summary.reached = "stone";

    // 5) STONE PICKAXE (need 3 cobble + 2 sticks; ensure sticks)
    log("STEP 5: crafting stone pickaxe");
    if (invCount(bot, mcData, "stick") < 2) {
      try { await craft(bot, mcData, "stick", 1, null); } catch (e) { /* need more planks maybe */ }
    }
    { const t = await ensureTableNearby(bot, mcData);
      await craft(bot, mcData, "stone_pickaxe", 1, t); }
    await ensureItemInHand(bot, "stone_pickaxe");
    summary.reached = "stone_pickaxe";

    // 6) COAL (fuel) + IRON ORE. These are buried — search nearby, else dig
    // deeper. Ore exposure grows as we tunnel, so digDownForBlocks doubles as
    // exploration; it collects whatever ore it happens to pass on the way.
    log("STEP 6: mining coal + iron ore");
    let gotCoal = await collect(bot, mcData, ["coal_ore", "deepslate_coal_ore"], 2, { verifyItems: ["coal"], maxDist: 20 });
    let fuelName = "coal";
    if (gotCoal < 1) {
      log("no coal nearby; will use planks as furnace fuel");
      fuelName = firstOwnedPlank() || "oak_planks";
    }
    let gotIron = await collect(bot, mcData, ["iron_ore", "deepslate_iron_ore"], 3, { verifyItems: ["raw_iron"], maxDist: 20 });
    if (gotIron < 3) {
      log(`only ${gotIron} raw_iron nearby; tunneling to expose more ore`);
      // dig down/around; collect any iron (and coal) we uncover
      gotIron += await digDownForBlocks(bot, mcData, ["raw_iron"], 3 - gotIron, 40);
      // opportunistically grab exposed iron via normal collect after tunneling
      if (gotIron < 3) {
        gotIron += await collect(bot, mcData, ["iron_ore", "deepslate_iron_ore"], 3 - gotIron, { verifyItems: ["raw_iron"], maxDist: 16 });
      }
    }
    if (gotIron < 1) throw new Error("no iron ore found");
    summary.reached = "iron_ore";

    // 7) SMELT — need 8 cobble for the furnace.
    log("STEP 7: smelting iron");
    let cobbleHave = invCount(bot, mcData, "cobblestone");
    if (cobbleHave < 8) {
      let more = await collect(bot, mcData, ["stone"], 8 - cobbleHave, { verifyItems: ["cobblestone"], maxDist: 12 });
      if (invCount(bot, mcData, "cobblestone") < 8) {
        await digDownForBlocks(bot, mcData, ["cobblestone"], 8 - invCount(bot, mcData, "cobblestone"), 15);
      }
    }
    { const t = await ensureTableNearby(bot, mcData);
      await craft(bot, mcData, "furnace", 1, t); }
    const furnace = await placeAndFind(bot, mcData, "furnace");
    await gotoWithTimeout(bot, new GoalGetToBlock(furnace.position.x, furnace.position.y, furnace.position.z), { timeoutMs: REACH_TIMEOUT_MS });
    const rawIronHave = invCount(bot, mcData, "raw_iron");
    await smeltIron(bot, mcData, furnace, fuelName, Math.min(rawIronHave, 3));
    summary.ironIngots = invCount(bot, mcData, "iron_ingot");
    summary.reached = "iron_ingot";

    // 8) IRON PICKAXE (3 iron + 2 sticks)
    log("STEP 8: crafting iron pickaxe");
    if (invCount(bot, mcData, "stick") < 2) {
      try { await craft(bot, mcData, "stick", 1, null); } catch (e) { /* */ }
    }
    { const t = await ensureTableNearby(bot, mcData);
      await craft(bot, mcData, "iron_pickaxe", 1, t); }
    await ensureItemInHand(bot, "iron_pickaxe");
    summary.ironPickaxe = true;
    summary.reached = "iron_pickaxe_DONE";
    log("✅ SURVIVAL CHAIN COMPLETE — iron pickaxe crafted!");
  } catch (err) {
    console.log(`[${bot.username}] ⛔ chain stopped at "${summary.reached}": ${err.message}`);
  } finally {
    stopAll(bot);
  }
  console.log(`[${bot.username}] 📊 summary: ${JSON.stringify(summary)}`);
  return summary;
}

/**
 * Single-bot from-scratch survival episode. Runs the whole chain inline in
 * entryPoint, then drives the stop-phase handshake (self-answered by the
 * loopback coordinator) to end + finalize recording.
 * @extends BaseEpisode
 */
class SurvivalIronPickEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    console.log(`[${bot.username}] 🚀 SurvivalIronPickEpisode entryPoint (episode ${episodeNum})`);
    try {
      await runSurvivalChain(bot, args);
    } catch (e) {
      console.log(`[${bot.username}] survival chain fatal: ${e.message}`);
    }

    // End the episode (loopback coordinator self-answers stopPhase -> stoppedPhase
    // -> episodeResolve; recording is finalized in base getOnStopPhaseFn).
    coordinator.onceEvent(
      "stopPhase",
      episodeNum,
      this.getOnStopPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        args.other_bot_name,
        episodeNum,
        args,
      ),
    );
    coordinator.sendToOtherBot(
      "stopPhase",
      bot.entity.position.clone(),
      episodeNum,
      "survivalIronPick end",
    );
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    await unequipHand(bot);
  }
}

module.exports = { SurvivalIronPickEpisode };
