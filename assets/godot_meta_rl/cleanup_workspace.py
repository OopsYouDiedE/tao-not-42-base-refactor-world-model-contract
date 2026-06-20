import os
import sys

# Get absolute path of this script's directory
workspace = os.path.dirname(os.path.abspath(__file__))
print(f"Workspace root: {workspace}")

# List of files we want to delete if they exist
files_to_delete = [
    "_async_min_godot.log",
    "_diag_godot.log",
    "_fps_capture.err",
    "_fps_nocap.err",
    "_ppo_run.out",
    "_train_ppo_2proc_godot.log",
    "_train_ppo_async_godot.log",
    "_train_ppo_godot.log",
    "ppo_spotlight_discrete.zip",
    "ppo_spotlight_discrete_2proc.zip",
    "ppo_spotlight_discrete_async.zip",
    "montage_10s.png",
    "montage_10s.png.import"
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
