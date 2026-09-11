#!/usr/bin/env python3
"""
Batch runner: executes 3 scripts across a grid of dropout and learning-rate
values, then organizes each run's output into a clearly-named folder.

Loop order: for each script -> for each dropout -> for each lr
Total runs = len(SCRIPTS) * len(DROPOUTS) * len(LEARNING_RATES)

Destination folder naming pattern:
    <DEST_ROOT>/<script_name>/dropout_<d>_lr_<lr>/

Each destination folder will contain:
    - stdout.log / stderr.log  (captured console output)
    - whatever files/folders your script produced (moved in automatically)

EDIT THE "CONFIG" SECTION BELOW to match your setup, then run:
    python batch_runner.py
"""

import subprocess
import shutil
import sys
from pathlib import Path
from itertools import product

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────────────────────

# Full paths to your three scripts
SCRIPTS = [
    Path("/path/to/scripts/train_baseline_cv_fold_test_added.py"),
    Path("/path/to/scripts/train_gated_cv_fold_test.py"),
    Path("/path/to/scripts/train_gated_fusion_cv_fold.py"),
]

DROPOUTS = [0.2, 0.3, 0.4]
LEARNING_RATES = [1e-3, 1e-4, 1e-5]

# ASSUMPTION: each script writes its output to a fixed folder called
# "output" inside its own working directory (a common convention).
# If your scripts write somewhere else, change this path.
# If your scripts instead accept an --output_dir flag, see the
# alternate `run_one` version further down (commented out) that
# passes the destination directly instead of moving files after.
DEFAULT_RUN_OUTPUT_DIR = Path("output")

# Where you want the organized results to end up
DEST_ROOT = Path("/path/to/batch_runs")

# Python interpreter to use (defaults to whatever runs this script)
PYTHON = sys.executable

# ─────────────────────────────────────────────────────────────


def run_one(script: Path, dropout: float, lr: float, dest_root: Path) -> bool:
    """Run a single script with given dropout/lr, save logs + outputs."""
    run_name = f"dropout_{dropout}_lr_{lr}"
    dest_dir = dest_root / script.stem / run_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(script),
        "--dropout", str(dropout),
        "--lr", str(lr),
    ]

    print(f"\n=== Running {script.name}  dropout={dropout}  lr={lr} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Always save console output, even on failure
    (dest_dir / "stdout.log").write_text(result.stdout)
    (dest_dir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) -- see {dest_dir / 'stderr.log'}")
        return False

    # Move the script's output folder contents into the destination
    if DEFAULT_RUN_OUTPUT_DIR.exists():
        for item in DEFAULT_RUN_OUTPUT_DIR.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        # Clean up so the next run starts with a fresh output folder
        shutil.rmtree(DEFAULT_RUN_OUTPUT_DIR, ignore_errors=True)
    else:
        print(f"  Warning: expected output folder '{DEFAULT_RUN_OUTPUT_DIR}' not found. "
              f"Only logs were saved.")

    print(f"  Saved to {dest_dir}")
    return True


# ─────────────────────────────────────────────────────────────
# ALTERNATE VERSION (uncomment + use instead of run_one above)
# if your scripts accept an --output_dir argument and write
# directly there themselves -- no moving needed.
# ─────────────────────────────────────────────────────────────
#
# def run_one(script: Path, dropout: float, lr: float, dest_root: Path) -> bool:
#     run_name = f"dropout_{dropout}_lr_{lr}"
#     dest_dir = dest_root / script.stem / run_name
#     dest_dir.mkdir(parents=True, exist_ok=True)
#
#     cmd = [
#         PYTHON, str(script),
#         "--dropout", str(dropout),
#         "--lr", str(lr),
#         "--output_dir", str(dest_dir),
#     ]
#
#     print(f"\n=== Running {script.name}  dropout={dropout}  lr={lr} ===")
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     (dest_dir / "stdout.log").write_text(result.stdout)
#     (dest_dir / "stderr.log").write_text(result.stderr)
#
#     if result.returncode != 0:
#         print(f"  FAILED (exit code {result.returncode})")
#         return False
#
#     print(f"  Saved to {dest_dir}")
#     return True


def main():
    total = len(SCRIPTS) * len(DROPOUTS) * len(LEARNING_RATES)
    done = 0
    failed = []

    for script in SCRIPTS:
        for dropout, lr in product(DROPOUTS, LEARNING_RATES):
            ok = run_one(script, dropout, lr, DEST_ROOT)
            done += 1
            print(f"Progress: {done}/{total}")
            if not ok:
                failed.append((script.name, dropout, lr))

    print("\n=== Done ===")
    if failed:
        print(f"{len(failed)} run(s) failed:")
        for name, d, lr in failed:
            print(f"  - {name} dropout={d} lr={lr}")
    else:
        print("All runs completed successfully.")


if __name__ == "__main__":
    main()