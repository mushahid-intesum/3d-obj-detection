#!/usr/bin/env python3
"""
pink_noise.py — Pink noise action sequence generator with probability integral transform.

Generates temporally-correlated action sequences that uniformly cover the action space.
Based on the exploration strategy from MINav (arXiv 2603.26441).
"""

import numpy as np
from scipy.stats import norm


def generate_pink_noise(length: int, beta: float = 1.0, seed: int = None) -> np.ndarray:
    """
    Generate a 1D pink noise signal in the time domain.

    Pink noise has a 1/f^β power spectrum. β=1 is classic pink noise.
    The output is zero-mean Gaussian-distributed with temporal correlation.

    Args:
        length: Number of samples to generate.
        beta:   Spectral exponent (1.0 = pink, 0.0 = white, 2.0 = brown).
        seed:   Random seed for reproducibility.

    Returns:
        np.ndarray of shape (length,), zero-mean Gaussian-distributed.
    """
    rng = np.random.default_rng(seed)

    # Generate in frequency domain
    freqs = np.fft.rfftfreq(length, d=1.0)
    # Avoid division by zero at DC
    freqs[0] = 1.0

    # Shape the power spectrum: amplitude ~ 1/f^(β/2)
    amplitudes = 1.0 / (freqs ** (beta / 2.0))
    amplitudes[0] = 0.0  # zero DC component (zero mean)

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, len(freqs))

    # Construct complex spectrum
    spectrum = amplitudes * np.exp(1j * phases)

    # Inverse FFT to time domain
    signal = np.fft.irfft(spectrum, n=length)

    return signal


def pink_noise_to_uniform(signal: np.ndarray) -> np.ndarray:
    """
    Apply the probability integral transform to convert Gaussian-distributed
    pink noise samples to Uniform(0, 1).

    The CDF Φ is monotonically increasing, so temporal ordering (and thus
    temporal correlation) is preserved.

    Args:
        signal: Gaussian-distributed noise signal.

    Returns:
        np.ndarray of same shape, values in [0, 1], uniformly distributed.
    """
    sigma = np.std(signal)
    if sigma < 1e-10:
        return np.full_like(signal, 0.5)
    return norm.cdf(signal / sigma)


def uniform_to_actions(uniform: np.ndarray, num_actions: int = 3) -> np.ndarray:
    """
    Map uniform [0, 1] values to discrete action indices.

    For 3 actions (forward, left, right):
        [0.0, 0.33) → 1 (turn_left)
        [0.33, 0.67) → 0 (forward)
        [0.67, 1.0]  → 2 (turn_right)

    Forward is the middle bin so the robot tends to go straight
    during smooth transitions through the center.

    Args:
        uniform:     Values in [0, 1].
        num_actions: Number of discrete action bins (default 3: fwd/left/right).

    Returns:
        np.ndarray of integer action indices.
    """
    # Bin edges
    bins = np.linspace(0, 1, num_actions + 1)

    # Map: left=1, forward=0, right=2
    # We reorder bins so that center maps to forward
    action_map = {0: 1, 1: 0, 2: 2}  # bin_idx → action_idx

    bin_indices = np.digitize(uniform, bins[1:])  # which bin
    bin_indices = np.clip(bin_indices, 0, num_actions - 1)

    actions = np.array([action_map.get(b, 0) for b in bin_indices])
    return actions


def generate_exploration_actions(
    length: int,
    beta: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """
    Full pipeline: pink noise → uniform transform → discrete actions.

    Args:
        length: Number of action steps to generate.
        beta:   Spectral exponent (1.0 = pink, 2.0 = brown/smoother).
        seed:   Random seed.

    Returns:
        np.ndarray of integer action indices: 0=forward, 1=left, 2=right.
    """
    raw = generate_pink_noise(length, beta=beta, seed=seed)
    uniform = pink_noise_to_uniform(raw)
    actions = uniform_to_actions(uniform, num_actions=3)
    return actions


def visualize_noise(length: int = 500, beta: float = 1.0, seed: int = 42):
    """
    Plot pink noise signal, its uniform transform, and the action histogram.
    Useful for debugging and understanding the exploration behavior.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    raw = generate_pink_noise(length, beta=beta, seed=seed)
    uniform = pink_noise_to_uniform(raw)
    actions = uniform_to_actions(uniform, num_actions=3)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Raw pink noise
    axes[0].plot(raw, linewidth=0.5, color='purple')
    axes[0].set_title(f'Pink Noise (β={beta}) — Gaussian distributed')
    axes[0].set_ylabel('Amplitude')

    # After CDF transform
    axes[1].plot(uniform, linewidth=0.5, color='teal')
    axes[1].set_title('After Φ transform — Uniform(0, 1)')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(-0.05, 1.05)

    # Discrete actions over time
    action_colors = {0: 'green', 1: 'blue', 2: 'red'}
    action_names = {0: 'Forward', 1: 'Left', 2: 'Right'}
    for t, a in enumerate(actions):
        axes[2].bar(t, 1, color=action_colors[a], width=1.0)
    axes[2].set_title('Discrete Actions Over Time')
    axes[2].set_ylabel('Action')
    axes[2].set_yticks([])

    # Action distribution histogram
    unique, counts = np.unique(actions, return_counts=True)
    bars = axes[3].bar([action_names[u] for u in unique],
                       counts / len(actions),
                       color=[action_colors[u] for u in unique])
    axes[3].set_title('Action Distribution (should be ~uniform)')
    axes[3].set_ylabel('Fraction')
    axes[3].axhline(y=1/3, color='gray', linestyle='--', label='ideal = 1/3')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('pink_noise_debug.png', dpi=150)
    plt.show()
    print("Saved: pink_noise_debug.png")


if __name__ == "__main__":
    # ─── Configuration ───
    LENGTH = 500        # Sequence length
    BETA   = 1.0        # Spectral exponent (1.0=pink, 2.0=brown)
    SEED   = 42         # Random seed

    # Print sample actions
    actions = generate_exploration_actions(LENGTH, beta=BETA, seed=SEED)
    names = {0: "FWD", 1: "LFT", 2: "RGT"}
    print(f"First 50 actions: {' '.join(names[a] for a in actions[:50])}")

    unique, counts = np.unique(actions, return_counts=True)
    print(f"\nDistribution over {LENGTH} steps:")
    for u, c in zip(unique, counts):
        print(f"  {names[u]}: {c} ({100*c/LENGTH:.1f}%)")

    # Visualize if matplotlib is available
    visualize_noise(LENGTH, BETA, SEED)
