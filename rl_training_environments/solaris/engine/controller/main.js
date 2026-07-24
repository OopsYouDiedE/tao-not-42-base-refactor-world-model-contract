/**
 * Modularized Minecraft Bot Coordination System
 *
 * This is the new modular version of controller.js with improved code organization.
 * All utility functions have been extracted into separate modules for better maintainability.
 */

const { parseArgs } = require("./config/args");
const { getOnSpawnFn } = require("./episodes-loop");
const { makeBot } = require("./utils/bot-factory");
const { BotCoordinator, LoopbackCoordinator } = require("./utils/coordination");
const { sleep } = require("./utils/helpers");

/**
 * Main function to initialize and run the bot
 */
async function main() {
  // Parse command line arguments
  const args = parseArgs();

  console.log(`Starting bot: ${args.bot_name}`);
  console.log(
    `Coordinator: ${args.bot_name}, Ports: ${args.coord_port}/${args.other_coord_port}`,
  );
  console.log(
    `[${args.bot_name}] Waiting ${args.bootstrap_wait_time} seconds before creating bot...`,
  );

  // Wait for bootstrap time
  await sleep(args.bootstrap_wait_time * 1000);

  // Create bot instance
  const bot = makeBot({
    username: args.bot_name,
    host: args.host,
    port: args.port,
    version: args.mc_version,
  });

  // Initialize shared RNG and coordinator.
  // Single-bot runs use a loopback coordinator that self-answers every phase
  // handshake (no second bot to ping-pong with); two-bot runs use the real
  // TCP peer coordinator unchanged.
  const CoordinatorClass = args.single_bot
    ? LoopbackCoordinator
    : BotCoordinator;
  const coordinator = new CoordinatorClass(
    args.bot_name,
    args.coord_port,
    args.other_coord_host,
    args.other_coord_port,
  );
  console.log(
    `[${args.bot_name}] using ${CoordinatorClass.name} (single_bot=${!!args.single_bot})`,
  );

  // Set up spawn event handler
  bot.once(
    "spawn",
    getOnSpawnFn(
      bot,
      args.act_recorder_host,
      args.act_recorder_port,
      coordinator,
      args,
    ),
  );

  // Handle system chat packets
  bot._client.on("packet", (data, meta) => {
    if (meta.name === "system_chat" && data && data.content) {
      console.log("SYSTEM:", JSON.stringify(data.content));
    }
  });
}

// Run the main function
main().catch(console.error);
