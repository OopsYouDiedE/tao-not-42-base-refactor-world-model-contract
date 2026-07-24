"""solaris 渲染器验收:逐帧动作 JSON + 录像 mp4 → 每个动作起止帧截图。

对外接口:
    python -m rl_training_environments.solaris.acceptance_boundary_shots
        --json <逐帧动作.json> --mp4 <同一序列的录像.mp4>
        --out <截图输出目录> [--contact-sheet]

输入是 solaris engine act_recorder 输出的"每帧一条动作记录 + 同序列 mp4 录像":
每帧 pos 含 action 子字典(布尔按键 +
camera:[dx,dy]),frame_count 与 mp4 帧严格 1:1 对齐。本模块把"动作"定义为:
  - 布尔键:值 False→True 记为该键一次动作的“开始帧”,True→False 记为“结束帧”;
  - camera:模长 0→非0 记为转头开始,非0→0 记为转头结束。
对每个动作实例,从 mp4 抽出其开始帧与结束帧存为 png(文件名带动作名/序号/frame_count),
可选生成 contact sheet 便于一眼核验。纯后处理,不依赖采集侧代码。所有产物落在
调用方指定目录(约定放在 gitignore 的 runs/ 下)。
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _camera_active(action: dict) -> bool:
    """相机是否在动:camera=[dx,dy] 任一非零即视为转头中。"""
    cam = action.get("camera", [0, 0])
    return bool(cam) and (abs(cam[0]) > 1e-6 or abs(cam[1]) > 1e-6)


def _action_state(action: dict) -> dict:
    """把一帧 action 归一成 {动作名: bool} 的激活状态。

    Args:
        action: 单帧记录里的 action 子字典。

    Returns:
        动作名 → 是否激活;camera 归一为单一 "camera" 通道。
    """
    state = {}
    for key, value in action.items():
        if key == "camera":
            state["camera"] = _camera_active(action)
        else:
            state[key] = bool(value)
    return state


def detect_boundaries(frames: list[dict]) -> list[dict]:
    """检测每个动作的开始/结束帧。

    Args:
        frames: 逐帧记录列表(每项含 action 子字典与 frame_count)。

    Returns:
        事件列表,每项 {action, edge('start'|'end'), frame_count, index, camera}。
        edge=start 表示该帧动作由灭变亮,end 表示由亮变灭。
    """
    events: list[dict] = []
    previous: dict[str, bool] = {}
    for index, frame in enumerate(frames):
        action = frame.get("action", {})
        current = _action_state(action)
        frame_count = int(frame.get("frame_count", index))
        for name, active in current.items():
            was = previous.get(name, False)
            if active and not was:
                events.append({
                    "action": name, "edge": "start", "frame_count": frame_count,
                    "index": index, "camera": action.get("camera", [0, 0]),
                })
            elif was and not active:
                events.append({
                    "action": name, "edge": "end", "frame_count": frame_count,
                    "index": index, "camera": action.get("camera", [0, 0]),
                })
        previous = current
    # 序列结束时仍激活的动作补一个 end(收尾帧)
    if frames:
        last_index = len(frames) - 1
        last_fc = int(frames[-1].get("frame_count", last_index))
        for name, active in previous.items():
            if active:
                events.append({
                    "action": name, "edge": "end", "frame_count": last_fc,
                    "index": last_index, "camera": frames[-1]["action"].get("camera", [0, 0]),
                })
    return events


def read_all_frames(mp4_path: Path) -> list[np.ndarray]:
    """把 mp4 全部帧读进内存,索引与逐帧记录的 frame_count 对齐。"""
    capture = cv2.VideoCapture(str(mp4_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    """在帧左上角烧入动作标签,便于截图脱离文件名也能辨认。"""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="动作起止帧截图")
    parser.add_argument("--json", required=True, help="逐帧动作 json")
    parser.add_argument("--mp4", required=True, help="同一序列的录像 mp4")
    parser.add_argument("--out", required=True, help="截图输出目录")
    parser.add_argument("--contact-sheet", action="store_true", help="额外生成核验拼图")
    args = parser.parse_args()

    frames_meta = json.loads(Path(args.json).read_text())
    video_frames = read_all_frames(Path(args.mp4))
    events = detect_boundaries(frames_meta)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"json 帧数={len(frames_meta)}  mp4 帧数={len(video_frames)}  动作边界事件={len(events)}")
    if len(video_frames) == 0:
        raise SystemExit("mp4 解码为 0 帧,录像无效")

    saved = []
    for order, event in enumerate(events):
        frame_index = event["frame_count"]
        if frame_index >= len(video_frames):
            frame_index = len(video_frames) - 1  # 收尾 end 可能落在最后一帧之外
        label = f"{event['action']}:{event['edge']}"
        if event["action"] == "camera":
            label += f" cam={event['camera']}"
        shot = annotate(video_frames[frame_index], f"{label} @f{frame_index}")
        name = f"{order:03d}_{event['action']}_{event['edge']}_f{frame_index:05d}.png"
        cv2.imwrite(str(out_dir / name), shot)
        saved.append((name, label, frame_index))

    print(f"已保存 {len(saved)} 张动作起止截图到 {out_dir}")
    for name, label, frame_index in saved:
        print(f"  {name}  ({label})")

    if args.contact_sheet and saved:
        shots = [cv2.imread(str(out_dir / name)) for name, _, _ in saved]
        columns = 4
        rows = (len(shots) + columns - 1) // columns
        height, width = shots[0].shape[:2]
        sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
        for i, shot in enumerate(shots):
            row, column = divmod(i, columns)
            sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = shot
        cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)
        print(f"核验拼图: {out_dir / 'contact_sheet.png'}")

    # 机器可读汇总,便于验收判定
    summary = {
        "json_frames": len(frames_meta),
        "mp4_frames": len(video_frames),
        "boundary_events": len(events),
        "actions_seen": sorted({event["action"] for event in events}),
        "shots": [{"file": name, "label": label, "frame": frame_index}
                  for name, label, frame_index in saved],
    }
    (out_dir / "boundary_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"汇总: {out_dir / 'boundary_summary.json'}")


if __name__ == "__main__":
    main()
