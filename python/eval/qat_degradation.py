
import os
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TeacherModel
from student_model import StudentModel


DATASET_DIR         = "./data/offline_dataset"
TEACHER_CKPT        = "./checkpoints/teacher/best.pt"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ACTIONS         = 4
BATCH_SIZE          = 256

FP32_CKPT           = "./checkpoints/student/best.pt"
QAT_CKPT            = None

TFLITE_ENCODER      = None
TFLITE_POLICY       = None


def load_teacher(path):
    teacher = TeacherModel(feat_dim=512, cue_dim=512,
                            num_actions=NUM_ACTIONS).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_student(path):
    student = StudentModel(num_actions=NUM_ACTIONS).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    return student


def load_eval_batch():
    try:
        from dataset import get_dataloader
        loader = get_dataloader(DATASET_DIR, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)
        return next(iter(loader))
    except Exception:
        return {
            "obs": torch.randn(BATCH_SIZE, 3, 96, 96),
            "goal": torch.randn(BATCH_SIZE, 3, 96, 96),
            "obs_proximity": torch.rand(BATCH_SIZE),
        }


@torch.no_grad()
def evaluate_pytorch_model(teacher, student, batch):
    obs = batch["obs"].to(DEVICE)
    goal = batch["goal"].to(DEVICE)
    prox = batch["obs_proximity"].to(DEVICE)

    teacher_cue = teacher.encode_and_correlate(obs, goal)
    teacher_logits = teacher.policy(teacher_cue, prox)
    teacher_actions = teacher_logits.argmax(dim=-1)

    student_logits = student(obs, goal, prox)
    student_actions = student_logits.argmax(dim=-1)

    agreement = (teacher_actions == student_actions).float().mean().item()

    teacher_soft = F.softmax(teacher_logits, dim=-1)
    student_log_soft = F.log_softmax(student_logits, dim=-1)
    kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean").item()

    return {"agreement": agreement, "kl_divergence": kl}


def evaluate_tflite_model(teacher, batch):
    try:
        import tensorflow as tf
    except ImportError:
        print("[!] TensorFlow not installed, skipping TFLite evaluation")
        return None

    enc_interp = tf.lite.Interpreter(model_path=TFLITE_ENCODER)
    enc_interp.allocate_tensors()
    pol_interp = tf.lite.Interpreter(model_path=TFLITE_POLICY)
    pol_interp.allocate_tensors()

    enc_input = enc_interp.get_input_details()
    enc_output = enc_interp.get_output_details()
    pol_input = pol_interp.get_input_details()
    pol_output = pol_interp.get_output_details()

    obs = batch["obs"].to(DEVICE)
    goal = batch["goal"].to(DEVICE)
    prox = batch["obs_proximity"]

    with torch.no_grad():
        teacher_cue = teacher.encode_and_correlate(obs, goal)
        teacher_logits = teacher.policy(teacher_cue, prox.to(DEVICE))
        teacher_actions = teacher_logits.argmax(dim=-1).cpu().numpy()

    B = obs.shape[0]
    tflite_actions = np.zeros(B, dtype=np.int32)
    obs_np = obs.cpu().numpy()
    goal_np = goal.cpu().numpy()
    prox_np = prox.numpy()

    for i in range(B):
        enc_scale = enc_input[0]["quantization"][0]
        enc_zp = enc_input[0]["quantization"][1]

        if enc_scale > 0:
            obs_q = np.clip(obs_np[i:i+1] / enc_scale + enc_zp,
                            -128, 127).astype(np.int8)
        else:
            obs_q = obs_np[i:i+1].astype(np.float32)
        enc_interp.set_tensor(enc_input[0]["index"], obs_q)
        enc_interp.invoke()
        obs_feat = enc_interp.get_tensor(enc_output[0]["index"])

        if enc_scale > 0:
            goal_q = np.clip(goal_np[i:i+1] / enc_scale + enc_zp,
                             -128, 127).astype(np.int8)
        else:
            goal_q = goal_np[i:i+1].astype(np.float32)
        enc_interp.set_tensor(enc_input[0]["index"], goal_q)
        enc_interp.invoke()
        goal_feat = enc_interp.get_tensor(enc_output[0]["index"])

        out_scale = enc_output[0]["quantization"][0]
        out_zp = enc_output[0]["quantization"][1]
        if out_scale > 0:
            obs_f = (obs_feat.astype(np.float32) - out_zp) * out_scale
            goal_f = (goal_feat.astype(np.float32) - out_zp) * out_scale
        else:
            obs_f = obs_feat.astype(np.float32)
            goal_f = goal_feat.astype(np.float32)

        obs_flat = obs_f.reshape(-1, obs_f.shape[-1])
        goal_flat = goal_f.reshape(-1, goal_f.shape[-1])
        obs_norm = obs_flat / (np.linalg.norm(obs_flat, axis=-1,
                                               keepdims=True) + 1e-8)
        goal_norm = goal_flat / (np.linalg.norm(goal_flat, axis=-1,
                                                 keepdims=True) + 1e-8)
        corr = (obs_norm @ goal_norm.T).flatten()

        h = int(np.sqrt(obs_norm.shape[0]))
        left_score = (obs_norm[:h] * goal_norm[:h]).sum(-1).mean()
        right_score = (obs_norm[-h:] * goal_norm[-h:]).sum(-1).mean()

        cue = np.concatenate([corr, [left_score, right_score, prox_np[i]]])

        pol_scale = pol_input[0]["quantization"][0]
        pol_zp = pol_input[0]["quantization"][1]
        if pol_scale > 0:
            cue_q = np.clip(cue / pol_scale + pol_zp,
                            -128, 127).astype(np.int8)
        else:
            cue_q = cue.astype(np.float32)
        pol_interp.set_tensor(pol_input[0]["index"], cue_q.reshape(1, -1))
        pol_interp.invoke()
        logits = pol_interp.get_tensor(pol_output[0]["index"])
        tflite_actions[i] = np.argmax(logits)

    agreement = (tflite_actions == teacher_actions).mean()
    return {"agreement": float(agreement), "kl_divergence": float("nan")}


def print_results(results):
    print(f"\n{'=' * 60}")
    print(f"  QAT DEGRADATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  {'Stage':<30} {'Agreement':>12} {'KL Div':>12}")
    print(f"  {'-' * 55}")
    for stage, metrics in results.items():
        agree = f"{metrics['agreement']:.1%}"
        kl = (f"{metrics['kl_divergence']:.4f}"
              if not np.isnan(metrics['kl_divergence']) else "N/A")
        print(f"  {stage:<30} {agree:>12} {kl:>12}")
    print(f"{'=' * 60}")

    print("\n% ─── LaTeX Table ───")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Action agreement with teacher across compression stages.}")
    print(r"\label{tab:qat}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Stage} & \textbf{Agreement (\%)} & \textbf{KL Div} \\")
    print(r"\midrule")
    for stage, metrics in results.items():
        agree = f"{metrics['agreement'] * 100:.1f}"
        kl = (f"{metrics['kl_divergence']:.4f}"
              if not np.isnan(metrics['kl_divergence']) else "---")
        print(f"{stage} & {agree} & {kl} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    teacher = load_teacher(TEACHER_CKPT)
    batch = load_eval_batch()
    results = OrderedDict()

    if FP32_CKPT and os.path.exists(FP32_CKPT):
        print("Evaluating FP32 student...")
        fp32 = load_student(FP32_CKPT)
        results["Post-distillation (FP32)"] = evaluate_pytorch_model(
            teacher, fp32, batch)

    if QAT_CKPT and os.path.exists(QAT_CKPT):
        print("Evaluating QAT student...")
        qat = load_student(QAT_CKPT)
        results["Post-QAT (fake INT8)"] = evaluate_pytorch_model(
            teacher, qat, batch)

    if TFLITE_ENCODER and TFLITE_POLICY:
        if os.path.exists(TFLITE_ENCODER) and os.path.exists(TFLITE_POLICY):
            print("Evaluating TFLite INT8...")
            tflite_res = evaluate_tflite_model(teacher, batch)
            if tflite_res:
                results["Exported TFLite (INT8)"] = tflite_res

    if results:
        print_results(results)
    else:
        print("[!] No checkpoints found. Set paths in the config section.")
