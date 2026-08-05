"""控制台页面。

页面按实例动态成列：顶部是 id 与初始化属性，下面左侧是视频窗口、右侧是控制与统计台，
最下方是动作序列输入。所有状态都从 `/api/instances` 拉取，页面本身不保存环境事实。
"""

from __future__ import annotations

import html
import json

_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: #14161a; color: #e6e8ea;
  font: 14px/1.5 "Segoe UI", "Noto Sans CJK SC", sans-serif; }
h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
.meta { color: #8b939c; font-size: 12px; margin-bottom: 20px; }
.meta code { color: #b9c2cc; }
.instance { border: 1px solid #262a30; border-radius: 8px; margin-bottom: 20px;
  background: #191c21; overflow: hidden; }
.head { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  padding: 12px 16px; background: #1e2228; border-bottom: 1px solid #262a30; }
.head .id { font-size: 15px; font-weight: 600; }
.head .attr { color: #8b939c; font-size: 12px; }
.head .attr b { color: #b9c2cc; font-weight: 500; }
.body { display: grid; grid-template-columns: minmax(320px, 1.4fr) 1fr; gap: 16px; padding: 16px; }
@media (max-width: 900px) { .body { grid-template-columns: 1fr; } }
.video { background: #000; border: 1px solid #262a30; border-radius: 6px;
  display: flex; align-items: center; justify-content: center; min-height: 200px; }
.video img { width: 100%; display: block; image-rendering: pixelated; }
.player { display: flex; flex-direction: column; gap: 8px; }
.scrubber input[type=range] { flex: 1; min-width: 120px; accent-color: #2b6cb0; }
.scrubber .attr { color: #8b939c; font-size: 12px; min-width: 120px; }
.console { display: flex; flex-direction: column; gap: 12px; }
fieldset { border: 1px solid #262a30; border-radius: 6px; margin: 0; padding: 10px 12px; }
legend { color: #8b939c; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  padding: 0 4px; }
label { display: inline-flex; align-items: center; gap: 5px; margin-right: 12px; font-size: 13px; }
input[type=number], select { background: #14161a; color: #e6e8ea;
  border: 1px solid #333941; border-radius: 4px; padding: 3px 6px; font: inherit; }
input[type=number] { width: 76px; }
button { background: #2b6cb0; color: #fff; border: 0; border-radius: 4px;
  padding: 6px 14px; font: inherit; cursor: pointer; }
button.ghost { background: #262a30; }
button:disabled { opacity: .45; cursor: default; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
table.stats { width: 100%; border-collapse: collapse; font-size: 12px; }
table.stats td { padding: 2px 0; border-bottom: 1px solid #22262c; }
table.stats td:last-child { text-align: right; font-variant-numeric: tabular-nums;
  color: #b9c2cc; }
.submit { padding: 0 16px 16px; }
textarea { width: 100%; min-height: 76px; background: #14161a; color: #e6e8ea;
  border: 1px solid #333941; border-radius: 6px; padding: 8px 10px; resize: vertical;
  font: 13px/1.5 "Cascadia Mono", "Consolas", monospace; }
.error { color: #f2777a; font-size: 12px; min-height: 16px; }
.badge { font-size: 11px; padding: 1px 7px; border-radius: 10px; background: #262a30;
  color: #8b939c; }
.badge.on { background: #2b6cb0; color: #fff; }
"""

_SCRIPT = """
const STATE = window.__BOOTSTRAP__;
const FIELDS = [
  ["submitted_sequences", "已提交序列"], ["submitted_ticks", "已提交 tick"],
  ["executed_ticks", "已执行 tick"], ["overrun_ticks", "队列空后续跑 tick"],
  ["observe_ticks", "观察次数"], ["expired_ticks", "过期 tick"],
  ["overwritten_ticks", "被覆盖 tick"], ["total_reward", "累计奖励"],
  ["mean_step_elapsed_ms", "平均 step (ms)"],
];

function h(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

async function call(slot, path, body) {
  const response = await fetch(`/api/instances/${slot}/${path}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({detail: response.statusText}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

function buildInstance(instance) {
  const slot = instance.slot;
  const error = h("div", {class: "error", role: "status", "aria-live": "polite"});
  const stats = h("tbody");
  const tick = h("span", {class: "attr"});
  const badge = h("span", {class: "badge"}, "空闲");

  const underflow = h("select", {"data-testid": `underflow-${slot}`}, ...[
    ["wait", "等待"], ["noop", "NoOp"], ["repeat_last", "重复上一动作"],
  ].map(([value, text]) => h("option", {value}, text)));
  underflow.value = instance.underflow;

  const budget = h("input", {type: "number", min: "0", step: "1",
    value: instance.max_overrun_ticks ?? 0, "data-testid": `overrun-budget-${slot}`});
  const unlimited = h("input", {type: "checkbox", "data-testid": `unlimited-${slot}`});
  unlimited.checked = instance.unlimited_overrun;

  const resetSnapshot = h("button", {class: "ghost", onclick: async () => {
    await runCommand("reset", {world: false});
  }}, "重置到快照");
  const resetWorld = h("button", {class: "ghost", onclick: async () => {
    await runCommand("reset", {world: true});
  }}, "重开世界");

  const video = h("img", {alt: `槽位 ${slot} 画面`});
  const scrub = h("input", {type: "range", min: "0", max: "0", value: "0", disabled: ""});
  const scrubLabel = h("span", {class: "attr"}, "实时");
  const live = h("button", {class: "ghost"}, "回到实时");
  let following = true;
  let frameCount = 0;

  function goLive() {
    following = true;
    scrubLabel.textContent = "实时";
    live.disabled = true;
    scrub.value = String(Math.max(0, frameCount - 1));
    video.src = `/api/instances/${slot}/stream`;
  }
  function showFrame(index) {
    following = false;
    live.disabled = false;
    // 断开 MJPEG 长连接，否则浏览器会继续用流覆盖回放帧。
    video.src = `/api/instances/${slot}/frame?index=${index}&r=${Date.now()}`;
    scrubLabel.textContent = `第 ${index + 1}/${frameCount} 帧`;
  }
  scrub.addEventListener("input", () => showFrame(Number(scrub.value)));
  live.addEventListener("click", goLive);

  async function pollFrames() {
    try {
      const response = await fetch(`/api/instances/${slot}/frames`);
      if (!response.ok) return;
      const info = await response.json();
      frameCount = info.count;
      scrub.disabled = frameCount === 0;
      scrub.max = String(Math.max(0, frameCount - 1));
      if (following) {
        scrub.value = String(Math.max(0, frameCount - 1));
        scrubLabel.textContent = frameCount
          ? `实时 · ${frameCount} 帧可回放` : "实时";
      }
    } catch (problem) { /* 网络抖动时保留上一状态 */ }
  }
  setInterval(pollFrames, 700);
  pollFrames();
  goLive();

  const sequence = h("textarea", {spellcheck: "false"});
  sequence.value = STATE.default_sequence;
  const send = h("button", {onclick: async () => {
    await runCommand("submit", {sequence: sequence.value});
  }}, "提交动作序列");
  const restore = h("button", {class: "ghost", onclick: () => {
    sequence.value = STATE.default_sequence;
  }}, "恢复默认序列");

  let currentState = instance;
  let stateEpoch = 0;
  let pollInFlight = false;
  let controlVersion = 0;
  let controlTimer = null;
  let controlQueue = Promise.resolve();
  let controlsDirty = false;
  let controlSaving = false;
  let commandPending = false;
  let uiError = "";

  function refreshDisabled() {
    const environmentLocked = currentState.running || commandPending;
    underflow.disabled = environmentLocked;
    unlimited.disabled = environmentLocked;
    budget.disabled = environmentLocked || unlimited.checked;

    const commandLocked = currentState.running || commandPending
      || controlsDirty || controlSaving;
    send.disabled = commandLocked;
    resetSnapshot.disabled = commandLocked;
    resetWorld.disabled = commandLocked;
    restore.disabled = commandPending;
  }

  function scheduleControl() {
    uiError = "";
    error.textContent = currentState.last_error || "";
    controlsDirty = true;
    stateEpoch += 1;
    const version = ++controlVersion;
    const body = {
      underflow: underflow.value,
      max_overrun_ticks: Number(budget.value),
      unlimited_overrun: unlimited.checked,
    };
    clearTimeout(controlTimer);
    refreshDisabled();
    controlTimer = setTimeout(() => {
      controlQueue = controlQueue.then(async () => {
        if (version !== controlVersion) return;
        controlSaving = true;
        refreshDisabled();
        try {
          const next = await call(slot, "control", body);
          if (version === controlVersion) {
            controlsDirty = false;
            uiError = "";
            render(next);
          }
        } catch (problem) {
          if (version === controlVersion) {
            controlsDirty = false;
            uiError = problem.message;
            await poll(true);
          }
        } finally {
          if (version === controlVersion) {
            controlSaving = false;
            refreshDisabled();
          }
        }
      });
    }, 150);
  }

  async function runCommand(path, body) {
    if (commandPending || controlsDirty || controlSaving || currentState.running) return;
    commandPending = true;
    stateEpoch += 1;
    uiError = "";
    error.textContent = currentState.last_error || "";
    refreshDisabled();
    try {
      const next = await call(slot, path, body);
      if (path === "submit") await poll(true);
      else render(next);
    } catch (problem) {
      uiError = problem.message;
      error.textContent = uiError;
    } finally {
      commandPending = false;
      refreshDisabled();
    }
  }

  underflow.addEventListener("change", scheduleControl);
  budget.addEventListener("input", scheduleControl);
  unlimited.addEventListener("change", scheduleControl);

  function render(next) {
    currentState = next;
    if (!controlsDirty) {
      underflow.value = next.underflow;
      unlimited.checked = next.unlimited_overrun;
      budget.value = String(next.max_overrun_ticks ?? 0);
    }
    tick.textContent = `tick ${next.current_tick} · 队列 ${next.buffered_ticks}`
      + ` · 快捷栏 ${next.selected_hotbar}`
      + (next.overrun_exhausted ? " · 队列已空且续跑预算用尽" : "");
    badge.textContent = next.running ? "执行中" : "空闲";
    badge.className = next.running ? "badge on" : "badge";
    error.textContent = uiError || next.last_error || "";
    stats.replaceChildren(...FIELDS.map(([key, text]) =>
      h("tr", {}, h("td", {}, text), h("td", {}, next.stats[key]))));
    if (next.stats.terminated || next.stats.truncated) {
      stats.append(h("tr", {}, h("td", {}, "环境结束"),
        h("td", {}, next.stats.terminated ? "terminated" : "truncated")));
    }
    refreshDisabled();
  }

  async function poll(force = false) {
    if (pollInFlight && !force) return;
    const epoch = stateEpoch;
    pollInFlight = true;
    try {
      const response = await fetch(`/api/instances/${slot}`);
      if (response.ok) {
        const next = await response.json();
        if (force || epoch === stateEpoch) render(next);
      }
    } catch (problem) { /* 网络抖动时保留上一帧状态 */ }
    finally { pollInFlight = false; }
  }
  setInterval(poll, 500);
  render(instance);

  const attrs = Object.entries(instance.initialization).map(([key, value]) =>
    h("span", {class: "attr"}, h("b", {}, key), ` ${value}`));

  return h("section", {class: "instance", "data-testid": `instance-${slot}`},
    h("div", {class: "head"},
      h("span", {class: "id"}, instance.instance_id), badge, tick, attrs),
    h("div", {class: "body"},
      h("div", {class: "player"},
        h("div", {class: "video"}, video),
        h("div", {class: "row scrubber"}, scrub, scrubLabel, live)),
        h("div", {class: "console"},
        h("fieldset", {}, h("legend", {}, "控制"),
          h("div", {class: "row"},
            h("label", {}, "队列耗尽策略", underflow),
            h("label", {"title": "队列耗尽后允许按下溢策略续跑的 tick 上限；队列内的空隙不计入"},
              "队列耗尽后续跑 tick", budget),
            h("label", {}, unlimited, "不限")),
          h("div", {class: "row"}, resetSnapshot, resetWorld)),
        h("fieldset", {}, h("legend", {}, "统计"),
          h("table", {class: "stats"}, stats)))),
    h("div", {class: "submit"},
      sequence,
      h("div", {class: "row"}, send, restore, error)));
}

document.getElementById("instances").replaceChildren(
  ...STATE.instances.map(buildInstance));
"""


def render_page(state: dict[str, object]) -> str:
    """把服务端状态嵌入页面；实例列表在浏览器侧按该状态动态生成。"""
    bootstrap = json.dumps(state, ensure_ascii=False)
    meta = (
        f"后端 <code>{html.escape(str(state['action_backend']))}</code> · "
        f"动作空间 <code>{html.escape(str(state['action_space']))}</code> · "
        f"runtime <code>{html.escape(str(state['runtime_version']))}</code> · "
        f"根快照 <code>{html.escape(str(state['root_snapshot']))}</code>"
    )
    return (
        "<!doctype html>\n"
        '<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>CraftGround 实例控制台</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>CraftGround 实例控制台</h1>"
        f'<div class="meta">{meta}</div>'
        '<div id="instances"></div>'
        f"<script>window.__BOOTSTRAP__ = {bootstrap};</script>"
        f"<script>{_SCRIPT}</script>"
        "</body></html>\n"
    )
