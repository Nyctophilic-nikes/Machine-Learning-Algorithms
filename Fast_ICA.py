import numpy as np
import matplotlib.pyplot as plt

#Generate a sinusoidal wave and a ramp wave. Plot them.
t = np.linspace(0, 1, 1000)
f = 5  # frequency
A = 1  # amplitude
sin_wave = A * np.sin(2 * np.pi * f * t)

# generating a ramp wave
ramp_wave = np.linspace(-1, 1, 1000)

# plot the waves
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(t, sin_wave)
axs[0].set_title("Sinusoidal Wave")
axs[1].plot(t, ramp_wave)
axs[1].set_title("Ramp Wave")
plt.tight_layout()
plt.show()

# create the mixing matrix
A = np.array([[0.5, 1], [1, 0.5]])

# mix the signals
mixed_signals = np.dot(A, np.array([sin_wave, ramp_wave]))

# plot the mixed signals
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(t, mixed_signals[0])
axs[0].set_title("Mixed Signal 1")
axs[1].plot(t, mixed_signals[1])
axs[1].set_title("Mixed Signal 2")
plt.tight_layout()
plt.show()

from sklearn.decomposition import FastICA

# perform ICA
ica = FastICA(n_components=2)
recovered_signals = ica.fit_transform(mixed_signals.T)

# transpose the recovered signals
recovered_signals = recovered_signals.T


# plot the recovered signals
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
axs[0].plot(t, recovered_signals[0])
axs[0].set_title("Recovered Signal 1")
axs[1].plot(t, recovered_signals[1])
axs[1].set_title("Recovered Signal 2")
plt.tight_layout()
plt.show()
