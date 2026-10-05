# %% [markdown]
# # MonoMH-Enhanced Training Notebook
#
# Run MonoMH-Enhanced training on **RunPod** or **Kaggle** with:
# - Configurable enhancement flags (A1, B2, C2, D1)
# - WandB experiment tracking
# - Automatic checkpointing & resume (survives session restarts)
#
# **To convert to .ipynb** (if needed):
# ```bash
# pip install jupytext
# jupytext --to notebook monomh_training.py
# ```

# %% [markdown]
# ## 1. Configuration
# **Edit the values below before running.** This is the only cell you need to modify.

# %%
# ═══════════════════════════════════════════════════════════════
#                    USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# ── WandB ──
WANDB_API_KEY = ""           # Paste your WandB API key here (leave empty for offline mode)
WANDB_PROJECT = "monomh-enhanced"
WANDB_RUN_NAME = "exp0_baseline"   # Give this run a descriptive name

# ── Enhancement Flags ──
ENABLE_A1_PROTOTYPE_FILTER = False       # Prototype-Based Hypothesis Filtering
ENABLE_B2_INTRINSIC_CONDITIONER = False  # Intrinsic-Conditioned Hypothesis Diversity
ENABLE_C2_DEPTH_PRIOR = False            # Depth Prior Warm-Start
ENABLE_D1_EARLY_EXIT = False             # Hypothesis Early Exit

# ── Training ──
MAX_EPOCHS = 200          # Total training epochs
BATCH_SIZE = 8            # Reduce to 4 if OOM on 12GB GPUs
NUM_WORKERS = 4           # Dataloader workers
EVAL_START = 10           # Start evaluation after this epoch
EVAL_FREQUENCY = 10       # Evaluate every N epochs
LEARNING_RATE = 0.00125

# ── Time Management ──
# Set a time limit (minutes) to gracefully stop before session expires.
# Kaggle GPU: ~540 min (9 hours). RunPod: depends on your budget.
# Set to 0 for no time limit.
TIME_LIMIT_MINUTES = 0    # 0 = no limit. Recommended: 510 for Kaggle, 0 for RunPod

# ── Dataset ──
# For KAGGLE: Upload KITTI as a dataset. Set the path where it appears.
# For RUNPOD: Set the path where KITTI is stored on your volume.
KITTI_PATH_OVERRIDE = ""  # Leave empty for auto-detection

# ── Repository ──
# Git URL of your MonoMH-Enhanced repository (for cloning on cloud)
REPO_URL = ""             # e.g., "https://github.com/youruser/MonoMH-Enhanced.git"
REPO_BRANCH = "main"

# ═══════════════════════════════════════════════════════════════

# %% [markdown]
# ## 2. Platform Detection & Path Setup

# %%
import os
import sys
import time
import signal
import subprocess
import shutil

# ── Detect platform ──
IS_KAGGLE = os.path.exists("/kaggle")
IS_RUNPOD = os.path.exists("/workspace") and not IS_KAGGLE
IS_COLAB = "COLAB_GPU" in os.environ
IS_LOCAL = not (IS_KAGGLE or IS_RUNPOD or IS_COLAB)

if IS_KAGGLE:
    PLATFORM = "Kaggle"
    PERSIST_DIR = "/kaggle/working"
    WORK_DIR = "/kaggle/working/monomh"
elif IS_RUNPOD:
    PLATFORM = "RunPod"
    PERSIST_DIR = "/workspace"
    WORK_DIR = "/workspace/monomh"
elif IS_COLAB:
    PLATFORM = "Colab"
    PERSIST_DIR = "/content/drive/MyDrive" if os.path.exists("/content/drive") else "/content"
    WORK_DIR = "/content/monomh"
else:
    PLATFORM = "Local"
    PERSIST_DIR = os.getcwd()
    WORK_DIR = os.path.join(os.getcwd(), "monomh")

# Directories
REPO_DIR = os.path.join(WORK_DIR, "MonoMH-Enhanced")
CHECKPOINT_DIR = os.path.join(PERSIST_DIR, "monomh_checkpoints", WANDB_RUN_NAME)
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs", WANDB_PROJECT, WANDB_RUN_NAME)

# Auto-detect KITTI path
if KITTI_PATH_OVERRIDE:
    KITTI_DIR = KITTI_PATH_OVERRIDE
elif IS_KAGGLE:
    # Common Kaggle dataset mount points — adjust to match your dataset name
    candidates = [
        "/kaggle/input/kitti-3d-object-detection",
        "/kaggle/input/kitti-3d",
        "/kaggle/input/kitti",
    ]
    KITTI_DIR = next((p for p in candidates if os.path.exists(p)), "/kaggle/input/kitti")
elif IS_RUNPOD:
    KITTI_DIR = "/workspace/kitti"
else:
    KITTI_DIR = os.path.join(WORK_DIR, "kitti")

print(f"╔══════════════════════════════════════════╗")
print(f"║  MonoMH-Enhanced Training Notebook       ║")
print(f"╠══════════════════════════════════════════╣")
print(f"║  Platform:    {PLATFORM:<27}║")
print(f"║  Work Dir:    {WORK_DIR:<27}║")
print(f"║  KITTI Dir:   {KITTI_DIR:<27}║")
print(f"║  Checkpoints: {CHECKPOINT_DIR:<27}║")
print(f"║  Output Dir:  {OUTPUT_DIR:<27}║")
print(f"╠══════════════════════════════════════════╣")
print(f"║  A1 Prototype Filter:   {'ON' if ENABLE_A1_PROTOTYPE_FILTER else 'OFF':<18}║")
print(f"║  B2 Intrinsic Cond:     {'ON' if ENABLE_B2_INTRINSIC_CONDITIONER else 'OFF':<18}║")
print(f"║  C2 Depth Prior:        {'ON' if ENABLE_C2_DEPTH_PRIOR else 'OFF':<18}║")
print(f"║  D1 Early Exit:         {'ON' if ENABLE_D1_EARLY_EXIT else 'OFF':<18}║")
print(f"╠══════════════════════════════════════════╣")
tlm = f"{TIME_LIMIT_MINUTES} min" if TIME_LIMIT_MINUTES > 0 else "No limit"
print(f"║  Time Limit:  {tlm:<27}║")
print(f"║  Max Epochs:  {MAX_EPOCHS:<27}║")
print(f"║  Batch Size:  {BATCH_SIZE:<27}║")
print(f"╚══════════════════════════════════════════╝")

# %% [markdown]
# ## 3. Clone / Setup Repository

# %%
os.makedirs(WORK_DIR, exist_ok=True)

if REPO_URL and not os.path.exists(REPO_DIR):
    print(f"Cloning repository from {REPO_URL}...")
    subprocess.run([
        "git", "clone", "--branch", REPO_BRANCH, "--depth", "1",
        REPO_URL, REPO_DIR
    ], check=True)
    print("✅ Repository cloned")
elif os.path.exists(REPO_DIR):
    print(f"✅ Repository already exists at {REPO_DIR}")
else:
    print("⚠️  No REPO_URL set and repo not found.")
    print(f"   Please manually place MonoMH-Enhanced at: {REPO_DIR}")
    print("   Or set REPO_URL in the Configuration cell.")

# Verify repo structure
required_files = ["tools/train_val.py", "lib/kitti.yaml", "lib/helpers/trainer_helper.py"]
for f in required_files:
    path = os.path.join(REPO_DIR, f)
    if os.path.exists(path):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} — MISSING")

# %% [markdown]
# ## 4. Install Dependencies

# %%
# Install from requirements.txt
req_path = os.path.join(REPO_DIR, "requirements.txt")
if os.path.exists(req_path):
    print(f"Installing from {req_path}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_path], check=True)
    print("✅ Installed from requirements.txt")
else:
    print("⚠️  requirements.txt not found — installing core dependencies manually...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "numpy", "opencv-python-headless",
        "Pillow", "matplotlib", "pyyaml", "tqdm", "wandb",
        "numba", "scikit-image"
    ], check=True)
    print("✅ Installed core dependencies")

# Verify GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("⚠️  No GPU detected! Training will be extremely slow.")

# %% [markdown]
# ## 5. Dataset Setup
# Verify KITTI dataset structure and generate ImageSets if missing.

# %%
# Check KITTI directory
print(f"Checking KITTI dataset at: {KITTI_DIR}")

kitti_ok = True
for subdir in ["training/image_2", "training/calib", "training/label_2"]:
    path = os.path.join(KITTI_DIR, subdir)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"  ✅ {subdir}/ ({count} files)")
    else:
        print(f"  ❌ {subdir}/ — NOT FOUND")
        kitti_ok = False

if not kitti_ok:
    print("\n⚠️  KITTI dataset is incomplete or not found!")
    print("    For Kaggle: Upload KITTI as a Dataset and update KITTI_PATH_OVERRIDE")
    print("    For RunPod: Place KITTI at /workspace/kitti or update KITTI_PATH_OVERRIDE")
else:
    # Generate ImageSets if missing
    imagesets_dir = os.path.join(KITTI_DIR, "ImageSets")
    if not os.path.isdir(imagesets_dir):
        print("\nGenerating ImageSets split files...")
        os.makedirs(imagesets_dir, exist_ok=True)

        train_img_dir = os.path.join(KITTI_DIR, "training", "image_2")
        all_ids = sorted([f.split('.')[0] for f in os.listdir(train_img_dir) if f.endswith('.png')])

        # Standard Chen et al. split: 0-3711 train, 3712-7480 val
        train_ids = [idx for idx in all_ids if int(idx) <= 3711]
        val_ids = [idx for idx in all_ids if int(idx) > 3711]

        test_dir = os.path.join(KITTI_DIR, "testing", "image_2")
        test_ids = sorted([f.split('.')[0] for f in os.listdir(test_dir) if f.endswith('.png')]) if os.path.isdir(test_dir) else []

        for name, ids in [("train", train_ids), ("val", val_ids), ("trainval", all_ids), ("test", test_ids)]:
            path = os.path.join(imagesets_dir, f"{name}.txt")
            with open(path, "w") as f:
                f.write("\n".join(ids) + "\n")
            print(f"  ✅ {name}.txt ({len(ids)} entries)")
    else:
        print(f"  ✅ ImageSets/ already exists")

    print("\n✅ KITTI dataset ready")

# %% [markdown]
# ## 6. Generate Training Configuration
# Creates `kitti.yaml` with your enhancement flags.

# %%
import yaml

config = {
    "dataset": {
        "type": "kitti",
        "root_dir": KITTI_DIR,
        "eval_cls": ["Car", "Pedestrian", "Cyclist"],
        "eval_dataset": "kitti",
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "class_merging": False,
        "use_dontcare": False,
        "use_3d_center": True,
        "writelist": ["Car", "Pedestrian", "Cyclist"],
        "random_flip": 0.5,
        "random_crop": 0.5,
        "scale": 0.4,
        "shift": 0.1,
    },
    "model": {
        "type": "MonoMH",
        "backbone": "dla34",
        "neck": "DLAUp",
    },
    "optimizer": {
        "type": "adam",
        "lr": LEARNING_RATE,
        "weight_decay": 0.00001,
    },
    "lr_scheduler": {
        "warmup": True,
        "decay_rate": 0.1,
        "decay_list": [120, 160],
    },
    "trainer": {
        "max_epoch": MAX_EPOCHS,
        "eval_start": EVAL_START,
        "eval_frequency": EVAL_FREQUENCY,
        "save_frequency": EVAL_FREQUENCY,
        "disp_frequency": 100,
    },
    "tester": {
        "tester_metrics": "kitti",
        "threshold": 0.2,
        "out_dir": None,
        "resume_model": None,
    },
    "enhancements": {
        "prototype_filter": {
            "enabled": ENABLE_A1_PROTOTYPE_FILTER,
            "num_prototypes": 192,
            "members_per_proto": 512,
            "similarity_threshold": 0.85,
            "depth_reliability_threshold": 0.3,
            "ema_alpha": 0.005,
            "feature_dim": 64,
            "warmup_epochs": 10,
        },
        "intrinsic_conditioner": {
            "enabled": ENABLE_B2_INTRINSIC_CONDITIONER,
            "base_threshold": 0.75,
            "modulation_range": 0.3,
            "hidden_dim": 32,
        },
        "depth_prior": {
            "enabled": ENABLE_C2_DEPTH_PRIOR,
            "prior_dir": "depth_prior",
            "warmstart_epochs": 50,
            "prior_loss_weight": 1.0,
        },
        "early_exit": {
            "enabled": ENABLE_D1_EARLY_EXIT,
            "easy_threshold": 0.2,
            "hard_threshold": 0.7,
            "hidden_dim": 64,
            "loss_weight": 0.1,
        },
    },
}

# Write config
config_path = os.path.join(REPO_DIR, "lib", "kitti_notebook.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Config written to {config_path}")
print()

# Pretty-print enhancement status
enh = config["enhancements"]
for name, cfg in enh.items():
    status = "✅ ON" if cfg["enabled"] else "   OFF"
    print(f"  {status}  {name}")

# %% [markdown]
# ## 7. WandB Login

# %%
import wandb

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_MODE"] = "online"
    print("✅ WandB logged in (online mode)")
else:
    os.environ["WANDB_MODE"] = "offline"
    print("⚠️  No WandB API key — running in offline mode")
    print("   Set WANDB_API_KEY in Configuration to enable cloud sync")

# %% [markdown]
# ## 8. Checkpoint Management
# Restore checkpoints from persistent storage if resuming.

# %%
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Link persistent checkpoint dir into the output directory
output_ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
if os.path.islink(output_ckpt_dir):
    os.unlink(output_ckpt_dir)

if os.path.isdir(output_ckpt_dir):
    # Move existing checkpoints to persistent storage
    for f in os.listdir(output_ckpt_dir):
        src = os.path.join(output_ckpt_dir, f)
        dst = os.path.join(CHECKPOINT_DIR, f)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    shutil.rmtree(output_ckpt_dir)

os.symlink(CHECKPOINT_DIR, output_ckpt_dir)

# Check for existing checkpoint
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    epoch = ckpt.get("epoch", -1)
    best = ckpt.get("best_results", {})
    best_m = best.get("best_m_result", -1)
    print(f"✅ Found checkpoint at epoch {epoch}")
    print(f"   Best Moderate AP3D: {best_m:.2f}")
    print(f"   Training will RESUME from epoch {epoch + 1}")
    del ckpt
else:
    print("ℹ️  No existing checkpoint — training will start from scratch")

# %% [markdown]
# ## 9. Training
# Runs training with automatic time management and checkpoint-based resume.

# %%
import subprocess
import signal
import time

# Build training command
train_cmd = [
    sys.executable, os.path.join(REPO_DIR, "tools", "train_val.py"),
    "--config", config_path,
    "--work-date", WANDB_PROJECT,
    "--work-dir", WANDB_RUN_NAME,
    "--save-path", os.path.join(WORK_DIR, "outputs"),
    "--resume",
]

env = os.environ.copy()
env["PYTHONPATH"] = REPO_DIR + ":" + env.get("PYTHONPATH", "")
env["WANDB_PROJECT"] = WANDB_PROJECT
env["WANDB_NAME"] = WANDB_RUN_NAME

print("=" * 60)
print("  STARTING TRAINING")
print("=" * 60)
print(f"  Command: {' '.join(train_cmd)}")
print(f"  CWD: {REPO_DIR}")
if TIME_LIMIT_MINUTES > 0:
    print(f"  Time limit: {TIME_LIMIT_MINUTES} minutes")
    print(f"  Will gracefully stop ~10 min before limit")
else:
    print(f"  Time limit: None")
print("=" * 60)

start_time = time.time()
process = None

def graceful_stop(signum, frame):
    """Handle SIGTERM/SIGINT — let current epoch finish, then stop."""
    global process
    print("\n⚠️  Received stop signal — finishing current epoch...")
    if process and process.poll() is None:
        process.send_signal(signal.SIGINT)

# Register signal handlers for platform shutdown warnings
signal.signal(signal.SIGTERM, graceful_stop)
signal.signal(signal.SIGINT, graceful_stop)

try:
    process = subprocess.Popen(
        train_cmd,
        cwd=REPO_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output while monitoring time
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

        # Check time limit
        if TIME_LIMIT_MINUTES > 0:
            elapsed_min = (time.time() - start_time) / 60
            remaining_min = TIME_LIMIT_MINUTES - elapsed_min
            if remaining_min <= 10:
                print(f"\n⏰ TIME LIMIT: {remaining_min:.1f} min remaining — stopping gracefully...")
                process.send_signal(signal.SIGINT)
                # Wait for graceful shutdown (current epoch checkpoint save)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.terminate()
                break

    process.wait()
    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 60)
    if process.returncode == 0:
        print(f"  ✅ Training completed successfully in {elapsed:.1f} minutes")
    elif process.returncode == -2:  # SIGINT
        print(f"  ⏸️  Training paused at {elapsed:.1f} minutes (checkpoint saved)")
        print(f"     Re-run this cell to resume from checkpoint")
    else:
        print(f"  ❌ Training exited with code {process.returncode} after {elapsed:.1f} minutes")
    print("=" * 60)

except Exception as e:
    elapsed = (time.time() - start_time) / 60
    print(f"\n❌ Error after {elapsed:.1f} minutes: {e}")
    if process and process.poll() is None:
        process.terminate()

# %% [markdown]
# ## 10. View Results

# %%
# Check for saved results
print("=" * 60)
print("  TRAINING RESULTS")
print("=" * 60)

# Check latest checkpoint
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    epoch = ckpt.get("epoch", -1)
    best = ckpt.get("best_results", {})
    print(f"\n  Last completed epoch: {epoch}")
    print(f"  Remaining epochs:    {max(0, MAX_EPOCHS - epoch)}")
    print()
    print(f"  Best AP₃D @ IoU 0.70:")
    print(f"    Easy:     {best.get('best_e_result', -1):.2f}  (epoch {best.get('best_e_epoch', -1)})")
    print(f"    Moderate: {best.get('best_m_result', -1):.2f}  (epoch {best.get('best_m_epoch', -1)})")
    print(f"    Hard:     {best.get('best_h_result', -1):.2f}  (epoch {best.get('best_h_epoch', -1)})")
    del ckpt
else:
    print("\n  No checkpoint found — training may not have completed an epoch yet.")

# List all saved checkpoints
print(f"\n  Saved checkpoints in {CHECKPOINT_DIR}:")
if os.path.isdir(CHECKPOINT_DIR):
    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
    for c in ckpts:
        size_mb = os.path.getsize(os.path.join(CHECKPOINT_DIR, c)) / 1e6
        print(f"    {c}  ({size_mb:.1f} MB)")
    if not ckpts:
        print("    (none)")

# Scan output logs for AP results
log_dir = OUTPUT_DIR
if os.path.isdir(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    for lf in log_files[:1]:
        print(f"\n  Latest log entries from {lf}:")
        with open(os.path.join(log_dir, lf)) as f:
            lines = f.readlines()
        # Show last few eval results
        eval_lines = [l.strip() for l in lines if "Best" in l or "3d@0.70" in l]
        for el in eval_lines[-6:]:
            print(f"    {el}")

print(f"\n{'=' * 60}")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    epoch = ckpt.get("epoch", -1)
    if epoch < MAX_EPOCHS:
        print(f"  ⏸️  Training incomplete ({epoch}/{MAX_EPOCHS} epochs)")
        print(f"     Re-run Cell 9 to continue from epoch {epoch + 1}")
    else:
        print(f"  ✅ Training complete ({epoch}/{MAX_EPOCHS} epochs)")
    del ckpt

# %% [markdown]
# ## 11. Download Checkpoints (Kaggle only)
# On Kaggle, checkpoints in `/kaggle/working/` are automatically saved as output.
# This cell creates a zip for easy download.

# %%
if IS_KAGGLE:
    import zipfile
    zip_path = os.path.join(PERSIST_DIR, "monomh_checkpoints.zip")
    print(f"Creating checkpoint archive at {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(CHECKPOINT_DIR):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, os.path.dirname(CHECKPOINT_DIR))
                zf.write(filepath, arcname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"✅ Archive created: {zip_path} ({size_mb:.1f} MB)")
    print("   This file will be available in your Kaggle notebook output.")
else:
    print(f"Checkpoints are persisted at: {CHECKPOINT_DIR}")
    print("No download needed on RunPod/local — files persist across sessions.")

# %% [markdown]
# ## 12. Quick Reference
#
# ### Resume after session timeout
# Simply re-run this notebook from **Cell 9 (Training)**. The checkpoint
# system will automatically detect `latest_checkpoint.pth` and resume
# from the last completed epoch.
#
# ### Run a different experiment
# 1. Change the enhancement flags in **Cell 1 (Configuration)**
# 2. Change `WANDB_RUN_NAME` to a new name (e.g., `exp1_A1_only`)
# 3. Re-run all cells from **Cell 6** onward
#
# ### Ablation matrix reference
#
# | Experiment | A1 | B2 | C2 | D1 | WANDB_RUN_NAME |
# |------------|:--:|:--:|:--:|:--:|----------------|
# | Baseline   | ❌ | ❌ | ❌ | ❌ | `exp0_baseline` |
# | A1 only    | ✅ | ❌ | ❌ | ❌ | `exp1_A1_only` |
# | B2 only    | ❌ | ✅ | ❌ | ❌ | `exp2_B2_only` |
# | C2 only    | ❌ | ❌ | ✅ | ❌ | `exp3_C2_only` |
# | D1 only    | ❌ | ❌ | ❌ | ✅ | `exp4_D1_only` |
# | A1 + D1    | ✅ | ❌ | ❌ | ✅ | `exp5_A1_D1` |
# | Full       | ✅ | ✅ | ✅ | ✅ | `exp6_full` |
