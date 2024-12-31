import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from contourpy.types import point_dtype
from numpy import cos, pi

def chirp_gen(duration, initial_f0 = 1000, mu_modifier = 550, fs = 44_100):
    """
    Creates different chirp sounds, depending on duration, initial_f0, and mu_modifier.
    Formula: cos(2pi * f_0 * t + 2pi * mu * t**2)
    """
    Ts = 1 / fs
    t = np.arange(0, duration, Ts)
    return t, cos(2 * pi * initial_f0 * t + 2 * pi * mu_modifier * t**2)


def point_seven_second_chirp():
    """
    Generates chirp from hardcoded values
    """
    # Generate and show 500 three times
    tt, sig = chirp_gen(0.7, 1_000, 550)
    start = slice(500)
    middle = slice(len(tt // 2) - 250, len(tt // 2) + 250)
    end = slice(-500, None)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    axes[0].plot(tt[start], sig[start])
    axes[0].set_title("Initial 500 Points")
    axes[0].set_xlabel(f"Time: {tt[0]:.5f}s to {tt[499]:.5f}s")
    axes[1].plot(tt[middle], sig[middle])
    axes[1].set_title("Middle 500 Points")
    axes[1].set_xlabel(f"Time: {tt[len(tt) // 2 - 250]:.5f}s to {tt[len(tt) // 2 + 249]:.5f}")
    axes[2].plot(tt[end], sig[end])
    axes[2].set_title("End 500 Points")
    axes[2].set_xlabel(f"Time: {tt[-500]:.5f}s to {tt[-1]:.5f}s")
    sd.play(sig, 44_100)
    sd.wait()
    plt.tight_layout()
    plt.show()


def chirp_with_noise():
    """
    Generates 20 seconds noise signal, with chirp added in the 12th second.
    """
    fs = 44_100
    noise_dur = 20
    chirp_len = 0.7

    tt = np.linspace(0, noise_dur, int(fs * noise_dur), endpoint=False)
    noise = np.random.randn(len(tt)) * 0.35

    __, sig = chirp_gen(chirp_len)
    start_idx = int(12 * fs)
    end_idx = start_idx + len(sig)

    noise[start_idx:end_idx] += sig

    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    # Full 20-second signal
    axes[0].plot(tt, noise, label="Noisy Signal with Embedded Chirp", color="blue")
    axes[0].axvline(12, color="red", linestyle="--", label="Chirp Start (12s)")
    axes[0].axvline(12 + chirp_len, color="green", linestyle="--", label="Chirp End (12.7s)")
    axes[0].set_title("Full 20-Second Noisy Signal with Chirp")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    axes[0].legend()

    # Zoomed-in view of the noisy segment
    zoom_start = start_idx
    zoom_end = end_idx
    axes[1].plot(tt[zoom_start:zoom_end], noise[zoom_start:zoom_end], label="Noisy Signal", alpha=0.8)
    axes[1].plot(tt[zoom_start:zoom_end], sig, label="Original Chirp", alpha=0.6, linestyle="--")
    axes[1].set_title("Zoomed-In View of Chirp (12s to 12.7s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    sd.play(noise, fs)
    sd.wait()


def find_chirp(noise_sig, chirp_ref, frame_size, step=100, fs=44_100):
    """
    Finds the reference chirp in the noisy signal and returns the start time index in seconds.
    """
    frame_size_samples = int(frame_size * fs)

    # Ensure chirp_ref matches frame size
    chirp_ref = chirp_ref[:frame_size_samples]
    chirp_ref_norm = np.linalg.norm(chirp_ref)

    max_similarity = 0
    best_index = 0

    for start_idx in range(0, len(noise_sig) - frame_size_samples + 1, step):
        current_frame = noise_sig[start_idx:start_idx + frame_size_samples]

        frame_norm = np.linalg.norm(current_frame)

        # Avoid division by zero
        if frame_norm == 0 or chirp_ref_norm == 0:
            continue

        # Compute the normalized inner product (correlation)
        similarity = np.dot(current_frame, chirp_ref) / (frame_norm * chirp_ref_norm)

        if similarity > max_similarity:
            max_similarity = similarity
            best_index = start_idx

    # Convert to seconds
    best_time = best_index / fs
    return best_time


def test_find_chirp_with_hidden_signal():
    """
    Tests find_chirp with a noise-only signal containing a hidden chirp.
    """
    fs = 44_100
    chirp_dur = 5
    noise_dur = 19
    start_time = 12

    _, chirp_ref = chirp_gen(chirp_dur, fs=fs)

    noisy_signal = np.random.randn(int(noise_dur * fs)) * 0.35

    embed_start_idx = int(start_time * fs)
    embed_end_idx = embed_start_idx + len(chirp_ref)
    noisy_signal[embed_start_idx:embed_end_idx] += chirp_ref

    frame_size = 0.1
    step = 100

    print("Detecting from generated chirp...")
    # Detect chirp location using find_chirp
    chirp_start_time = find_chirp(noisy_signal, chirp_ref, frame_size, step, fs)
    print(f"The chirp starts at {chirp_start_time:.2f} seconds.")

    time_noisy_signal = np.linspace(0, noise_dur, len(noisy_signal))

    play_and_display(chirp_start_time, fs, noisy_signal, time_noisy_signal)


def test_find_chirp_with_loaded_data():
    """
    Tests find_chirp with loaded sound data (chirp and noisy signal).
    """
    # Load the chirp and noisy signal
    chirp = np.load("chirp.npy")
    xnsig = np.load("xnsig.npy")

    fs = 44_100
    frame_size = 0.7
    step = 100

    print("Detecting from loaded chirp...")
    # Detect chirp location
    chirp_start_time = find_chirp(xnsig, chirp, frame_size, step, fs)
    print(f"The chirp starts at {chirp_start_time:.2f} seconds.")

    time_xnsig = np.linspace(0, len(xnsig) / fs, len(xnsig))

    play_and_display(chirp_start_time, fs, xnsig, time_xnsig, chirp)


def play_and_display(chirp_start_time, fs, noisy_signal, tt, chirp_ref=None):
    """
    Plays the noisy signal, displays it with detected chirp, and optionally shows cross-correlation with the maximum highlighted.
    """
    if chirp_ref is not None:
        print("Calculating cross-correlation...")
        # Compute cross-correlation
        correlation = np.correlate(noisy_signal, chirp_ref, mode="valid")
        correlation_time = np.linspace(0, len(correlation) / fs, len(correlation))

        # Find maximum correlation
        max_value = np.max(correlation)
        max_index = np.where(correlation == max_value)[0][0]
        max_time = correlation_time[max_index]  # First occurrence of the max

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot noisy signal
        plot_ax1(ax1, chirp_start_time, noisy_signal, tt)

        # Plot cross-correlation
        plot_ax2(ax2, correlation, correlation_time, max_time, max_value)
    else:
        # Create a single plot without cross-correlation
        fig, ax1 = plt.subplots(figsize=(12, 6))
        plot_ax1(ax1, chirp_start_time, noisy_signal, tt)

    plt.tight_layout()
    plt.show()

    # Play the noisy signal
    print("Playing the noisy signal...")
    sd.play(noisy_signal, fs)
    sd.wait()


def plot_ax1(ax1, chirp_start_time, noisy_signal, tt):
    """
    Plots the noisy signal
    """
    ax1.plot(tt, noisy_signal, label="Noisy Signal with Hidden Chirp")
    ax1.axvline(chirp_start_time, color='red', linestyle='--', label="Detected Chirp Start")
    ax1.set_title("Noisy Signal with Hidden Chirp")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()


def plot_ax2(ax2, correlation, correlation_time, max_time, max_value):
    """
    Plots the cross-correlation with highlighted. maximum
    """
    ax2.plot(correlation_time, correlation, label="Cross-Correlation Coefficient")
    ax2.scatter(max_time, max_value, color='red', label=f"Max Value ({max_value:.2f})", zorder=5)
    ax2.set_title("Cross-Correlation Between Noisy Signal and Chirp Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Correlation Coefficient")
    ax2.axvline(max_time, color='red', linestyle='--', label=f"Max Point at {max_time:.2f}s")
    ax2.grid(True)
    ax2.legend()


if __name__ == "__main__":
    print("0.7 chirp:")
    point_seven_second_chirp() # B
    print("20 seconds noise with chirp: ")
    chirp_with_noise() # D
    print("Generated chirp: ")
    test_find_chirp_with_hidden_signal() # E
    print("Loaded data: ")
    test_find_chirp_with_loaded_data() # F






