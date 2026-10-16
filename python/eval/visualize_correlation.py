
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TeacherModel
from student_model import StudentModel, CORR_POOL_SIZE


DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ACTIONS         = 4
ACTION_NAMES        = ["FWD", "RIGHT", "LEFT", "STOP"]

TEACHER_CKPT        = None
STUDENT_CKPT        = None

OBS_IMAGE           = None
GOAL_IMAGE          = None

PROXIMITY           = 1.0
OUTPUT_DIR          = "./figures"


def load_image(path, size=96):
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def extract_correlation_details(student, obs, goal):
    f_obs = student.encoder(obs.to(DEVICE))
    f_goal = student.encoder(goal.to(DEVICE))

    pool_size = CORR_POOL_SIZE
    f_obs_pool = F.adaptive_avg_pool2d(f_obs, pool_size)
    f_goal_pool = F.adaptive_avg_pool2d(f_goal, pool_size)

    B, C, H, W = f_obs_pool.shape
    obs_flat = f_obs_pool.view(B, C, -1).permute(0, 2, 1)
    goal_flat = f_goal_pool.view(B, C, -1).permute(0, 2, 1)

    obs_norm = F.normalize(obs_flat, dim=-1)
    goal_norm = F.normalize(goal_flat, dim=-1)

    corr_matrix = torch.bmm(obs_norm, goal_norm.transpose(1, 2))

    left_obs = obs_norm[:, :H, :]
    left_goal = goal_norm[:, :H, :]
    right_obs = obs_norm[:, -H:, :]
    right_goal = goal_norm[:, -H:, :]

    left_score = (left_obs * left_goal).sum(-1).mean().item()
    right_score = (right_obs * right_goal).sum(-1).mean().item()

    corr_cue = student.get_cue(obs.to(DEVICE), goal.to(DEVICE))

    return {
        "corr_matrix": corr_matrix[0].cpu().numpy(),
        "left_score": left_score,
        "right_score": right_score,
        "corr_cue": corr_cue[0].cpu().numpy(),
    }


@torch.no_grad()
def extract_teacher_cue(teacher, obs, goal):
    cue = teacher.encode_and_correlate(obs.to(DEVICE), goal.to(DEVICE))
    return cue[0].cpu().numpy()


def plot_correlation(details, teacher_cue=None, action_label=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Skipping visualization.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im = axes[0].imshow(details["corr_matrix"], cmap="RdBu_r",
                         vmin=-1, vmax=1, aspect="equal")
    axes[0].set_title("Student Correlation Matrix (16×16)")
    axes[0].set_xlabel("Goal positions")
    axes[0].set_ylabel("Observation positions")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    axes[1].bar(range(len(details["corr_cue"])), details["corr_cue"],
                color="#4A90D9", alpha=0.7, width=1.0)
    axes[1].set_title("Student Correlation Cue (258-dim)")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Value")
    axes[1].set_xlim(0, len(details["corr_cue"]))

    lr_labels = ["Left", "Right"]
    lr_vals = [details["left_score"], details["right_score"]]
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in lr_vals]
    axes[2].barh(lr_labels, lr_vals, color=colors)
    axes[2].set_title("Directional Scores")
    axes[2].set_xlim(-1, 1)
    axes[2].axvline(0, color="gray", linewidth=0.5)

    if action_label:
        fig.suptitle(f"Predicted action: {action_label}", fontsize=12, y=1.02)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[*] Saved: {path}")
    plt.close()

    if teacher_cue is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 5))

        axes[0].bar(range(len(teacher_cue)), teacher_cue,
                    color="#E67E22", alpha=0.7, width=1.0)
        axes[0].set_title("Teacher Cue (512-dim)")
        axes[0].set_ylabel("Value")

        axes[1].bar(range(len(details["corr_cue"])), details["corr_cue"],
                    color="#4A90D9", alpha=0.7, width=1.0)
        axes[1].set_title("Student Cue (258-dim)")
        axes[1].set_xlabel("Dimension")
        axes[1].set_ylabel("Value")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "teacher_vs_student_cue.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[*] Saved: {path}")
        plt.close()


if __name__ == "__main__":
    student = StudentModel(num_actions=NUM_ACTIONS).to(DEVICE)
    if STUDENT_CKPT and os.path.exists(STUDENT_CKPT):
        ckpt = torch.load(STUDENT_CKPT, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt["model_state"])
        print(f"[*] Student loaded from {STUDENT_CKPT}")
    else:
        print("[!] No student checkpoint, using random weights")
    student.eval()

    teacher = None
    teacher_cue = None
    if TEACHER_CKPT and os.path.exists(TEACHER_CKPT):
        teacher = TeacherModel(feat_dim=512, cue_dim=512,
                                num_actions=NUM_ACTIONS).to(DEVICE)
        ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(ckpt["model_state"])
        teacher.eval()
        print(f"[*] Teacher loaded from {TEACHER_CKPT}")

    if OBS_IMAGE and GOAL_IMAGE:
        obs = load_image(OBS_IMAGE)
        goal = load_image(GOAL_IMAGE)
    else:
        print("[!] No images provided, using synthetic data")
        obs = torch.randn(1, 3, 96, 96)
        goal = torch.randn(1, 3, 96, 96)

    details = extract_correlation_details(student, obs, goal)
    if teacher is not None:
        teacher_cue = extract_teacher_cue(teacher, obs, goal)

    prox = torch.tensor([PROXIMITY])
    with torch.no_grad():
        logits = student(obs.to(DEVICE), goal.to(DEVICE), prox.to(DEVICE))
        action = logits.argmax(dim=-1).item()
    action_label = ACTION_NAMES[action]

    print(f"\nCorrelation matrix range: [{details['corr_matrix'].min():.3f}, "
          f"{details['corr_matrix'].max():.3f}]")
    print(f"L/R scores: left={details['left_score']:.3f}, "
          f"right={details['right_score']:.3f}")
    print(f"Predicted action: {action_label}")

    plot_correlation(details, teacher_cue=teacher_cue,
                     action_label=action_label)
