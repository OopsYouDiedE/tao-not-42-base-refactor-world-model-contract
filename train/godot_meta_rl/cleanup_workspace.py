"""清理 Godot 工程目录里训练/诊断产生的临时日志与模型产物（不入库的中间文件）。

用法: python train/godot_meta_rl/cleanup_workspace.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.godot_rl import shared_mem_env as E

# 临时文件落在 Godot 工程目录（各脚本以 E.PROJECT_DIR 为基准写日志/模型）。
workspace = E.PROJECT_DIR
print(f"Workspace root: {workspace}")

files_to_delete = [
    "_async_min_godot.log",
    "_diag_godot.log",
    "_fps_capture.err",
    "_fps_nocap.err",
    "_ppo_run.out",
    "_train_smoke.log",
    "_train_ppo_2proc_godot.log",
    "_train_ppo_async_godot.log",
    "_train_ppo_godot.log",
    "_godot_test_run.log",
    "_godot_completeness.log",
    "_godot_compare_run.log",
    "_godot_mode_fixed.log",
    "_godot_mode_decoupled.log",
    "_learner_err.log",
    "ppo_spotlight_discrete.zip",
    "ppo_spotlight_discrete_2proc.zip",
    "ppo_spotlight_discrete_async.zip",
]

print("Scanning for temporary files...")
deleted_count = 0
for filename in files_to_delete:
    filepath = os.path.join(workspace, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Successfully deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {filename}: {e}", file=sys.stderr)
    else:
        print(f"File not found (already clean): {filename}")

print(f"Cleanup complete. Deleted {deleted_count} files.")
