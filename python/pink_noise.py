
import numpy as np
from scipy.stats import norm


def generate_pink_noise(length: int, beta: float = 1.0, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    freqs = np.fft.rfftfreq(length, d=1.0)
    freqs[0] = 1.0

    amplitudes = 1.0 / (freqs ** (beta / 2.0))
    amplitudes[0] = 0.0

    phases = rng.uniform(0, 2 * np.pi, len(freqs))

    spectrum = amplitudes * np.exp(1j * phases)

    signal = np.fft.irfft(spectrum, n=length)

    return signal


def pink_noise_to_uniform(signal: np.ndarray) -> np.ndarray:
    sigma = np.std(signal)
    if sigma < 1e-10:
        return np.full_like(signal, 0.5)
    return norm.cdf(signal / sigma)


def uniform_to_actions(uniform: np.ndarray, num_actions: int = 3) -> np.ndarray:
    bins = np.linspace(0, 1, num_actions + 1)

    action_map = {0: 1, 1: 0, 2: 2}

    bin_indices = np.digitize(uniform, bins[1:])
    bin_indices = np.clip(bin_indices, 0, num_actions - 1)

    actions = np.array([action_map.get(b, 0) for b in bin_indices])
    return actions


def generate_exploration_actions(
    length: int,
    beta: float = 1.0,
    seed: int = None
) -> np.ndarray:
    raw = generate_pink_noise(length, beta=beta, seed=seed)
    uniform = pink_noise_to_uniform(raw)
    actions = uniform_to_actions(uniform, num_actions=3)
    return actions


def visualize_noise(length: int = 500, beta: float = 1.0, seed: int = 42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping visualization")
        return

    raw = generate_pink_noise(length, beta=beta, seed=seed)
    uniform = pink_noise_to_uniform(raw)
    actions = uniform_to_actions(uniform, num_actions=3)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(raw, linewidth=0.5, color='purple')
    axes[0].set_title(f'Pink Noise (β={beta}) — Gaussian distributed')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(uniform, linewidth=0.5, color='teal')
    axes[1].set_title('After Φ transform — Uniform(0, 1)')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(-0.05, 1.05)

    action_colors = {0: 'green', 1: 'blue', 2: 'red'}
    action_names = {0: 'Forward', 1: 'Left', 2: 'Right'}
    for t, a in enumerate(actions):
        axes[2].bar(t, 1, color=action_colors[a], width=1.0)
    axes[2].set_title('Discrete Actions Over Time')
    axes[2].set_ylabel('Action')
    axes[2].set_yticks([])

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
    LENGTH = 500
    BETA   = 1.0
    SEED   = 42

    actions = generate_exploration_actions(LENGTH, beta=BETA, seed=SEED)
    names = {0: "FWD", 1: "LFT", 2: "RGT"}
    print(f"First 50 actions: {' '.join(names[a] for a in actions[:50])}")

    unique, counts = np.unique(actions, return_counts=True)
    print(f"\nDistribution over {LENGTH} steps:")
    for u, c in zip(unique, counts):
        print(f"  {names[u]}: {c} ({100*c/LENGTH:.1f}%)")

    visualize_noise(LENGTH, BETA, SEED)
