"""
Quick Script for Visualizing Dynamic Scenes
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-01-07

Heads up: This script should be run from the project root and is dependent on a hardcoded filepath.
"""

DYNAMICSCENES_NPY_FILEPATH = './frame_prediction/dynamic_scenes2d/dynamic_scenes2d_motionv01_64_10_1_100_pt00.npy'

# Visualize 50 frames from one generated scene (animation + grid fallback)
import os
import numpy as np

# Try to select a GUI backend suitable for Xorg windows. This must be done
# before importing `pyplot` in order to take effect.
try:
    import matplotlib
    # Prefer TkAgg, fallback to Qt5Agg
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Qt5Agg')
except Exception:
    # If matplotlib import fails, let the subsequent import raise the error
    pass

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ajuste o caminho do arquivo se necessário
file_path = DYNAMICSCENES_NPY_FILEPATH
scenes = np.load(file_path, allow_pickle=True)
# `scenes` foi salvo como lista; pegamos o primeiro cenário
scene = scenes[0]
# scene tem shape (n_frames, H, W)
n_frames = scene.shape[0]

# If there is no DISPLAY, opening an X window will fail — inform the user.
if 'DISPLAY' not in os.environ or not os.environ.get('DISPLAY'):
    print('Warning: DISPLAY environment variable is not set.\n'
          'To show an X window ensure you run this inside an X session (Xorg) or set DISPLAY.')

# Animation for X window: create the figure and show it via `plt.show()`.
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(scene[0], cmap='binary', vmin=0, vmax=1)
ax.axis('off')

def update(i):
    im.set_data(scene[i])
    ax.set_title(f'Frame {i+1}/{n_frames}')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)

# Mostrar em janela X (bloqueia até a janela ser fechada)
plt.show()

# Depois de fechar a janela, exibimos uma grade com os primeiros 50 frames
max_to_show = min(n_frames, 50)
rows, cols = 5, 10
fig2 = plt.figure(figsize=(12, 6))
for i in range(max_to_show):
    ax2 = fig2.add_subplot(rows, cols, i + 1)
    ax2.imshow(scene[i], cmap='binary', vmin=0, vmax=1)
    ax2.axis('off')
fig2.suptitle(f'First {max_to_show} frames')
plt.tight_layout()
plt.show()