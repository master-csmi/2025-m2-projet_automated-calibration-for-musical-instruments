import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

path = "../experiments/data_calib/Audio-Technica 0005 [2024-06-17 175153].aif"

audio, fs = sf.read(path)

print("Sampling rate :", fs, "Hz")
print("Shape         :", audio.shape)
print("Duration      :", len(audio) / fs, "s")
print("dtype         :", audio.dtype)

if audio.ndim == 1:
    print("Channels      : 1")
else:
    print("Channels      :", audio.shape[1])

print("Max abs       :", np.max(np.abs(audio)))
print("RMS           :", np.sqrt(np.mean(audio**2)))


# ------------------------------------------------------------
# RMS locale par fenêtres de 20 ms
# ------------------------------------------------------------

window_duration = 0.020
window_size = int(window_duration * fs)

n_windows = len(audio) // window_size

audio_cut = audio[:n_windows * window_size]

frames = audio_cut.reshape(n_windows, window_size)

rms = np.sqrt(np.mean(frames**2, axis=1))

t_rms = (
    np.arange(n_windows) * window_size
    + window_size / 2
) / fs

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

plt.figure(figsize=(18, 5))

plt.plot(t_rms, rms)

plt.xlabel("Time (s)")
plt.ylabel("Local RMS")
plt.title("RMS envelope of the AIF recording")
plt.grid()

plt.tight_layout()
plt.savefig("aif_rms_overview.png", dpi=150)
plt.close()

print("Figure saved: aif_rms_overview.png")